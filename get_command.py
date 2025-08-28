'''
get_command - Voice Command Recognition and Execution for Drone Control using MAVSDK
This script listens for voice commands, parses them, and executes corresponding drone commands using MAVSDK.

Before running this script, ensure the following:
1. Download and extract the Vosk model from: https://alphacephei.com/vosk/models
2. Install the required Python packages using the following command: pip install -r requirements.txt
'''
import json
import pyaudio
import re
import asyncio
import math
import time
from vosk import Model, KaldiRecognizer
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import PositionNedYaw, OffboardError

# Constants
SIM_MODE = True  # Sets simulation mode; should be False when running with Pixhawk. THIS IS THE ONLY VARIABLE THAT SHOULD BE ADJUSTED

# Pre-compiled regex patterns
DIGIT_PATTERN = re.compile(r'\d+')
SEPARATOR_PATTERN = re.compile(r'(?:,\s*(?:then\s+|and\s+|next\s+)?|,?\s+(?:then|and|next|after\s+that|followed\s+by|afterward)\s+)')

# Unit patterns compile 
UNIT_PATTERNS = {
    'inches': re.compile(r'\b(?:inch|inches)\b'),
    'feet': re.compile(r'\b(?:feet|foot)\b'), 
    'yards': re.compile(r'\b(?:yard|yards)\b'),
    'degrees': re.compile(r'\b(?:degree|degrees)\b')
}

# Command dictionary maps spoken commands to their corresponding action types
COMMAND_DICT = {
    "STOP":     ["stop", "stop drone"],
    "STANDBY":  ["standby", "stand by"],
    "LAND":     ["land", "land drone"],
    "FORWARD":  ["forward"],
    "BACKWARD": ["backward"],
    "UP":       ["up"],
    "DOWN":     ["down"],
    "LEFT":     ["left"],
    "RIGHT":    ["right"],
    "RO_LEFT":  ["turn left", "rotate left"],
    "RO_RIGHT": ["turn right", "rotate right"],
    "TAKEOFF":  ["takeoff", "take off"],
    "DISARM":   ["disarm", "disarm drone"],
    "ARM":      ["arm", "arm drone"],
}

# Command Lookup flattens command lookup table for O(1) access rather than needing nested loops
COMMAND_LOOKUP = {trigger: cmdType for cmdType, triggers in COMMAND_DICT.items() for trigger in triggers}

# Number dictionary converts spoken numbers to integer variables
NUM_DICT = {
    'one': 1, 'two': 2, 'to': 2, 'three': 3, 'four': 4, 'for': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100
}

# NED (North East Down) movement direction unit vectors
BASE_DIRECTION_OFFSETS = {
    "FORWARD": (1, 0, 0),     # Positive North
    "BACKWARD": (-1, 0, 0),   # Negative North
    "LEFT": (0, -1, 0),       # Negative East
    "RIGHT": (0, 1, 0),       # Positive East
    "UP": (0, 0, -1),         # Negative Down (up in NED)
    "DOWN": (0, 0, 1)         # Positive Down
}

# Operations Timeouts (seconds)
CONNECTION_TIMEOUT = 30.0   # Initial drone connection
ACTION_TIMEOUT = 10.0       # Basic actions (arm, disarm, takeoff, land)
MOVEMENT_TIMEOUT = 15.0     # Movement and rotation commands
OFFBOARD_TIMEOUT = 5.0      # Offboard mode start/stop operations

# Connection retry configuration
MAX_CONNECTION_RETRIES = 5      # Maximum number of connection retry attempts
INITIAL_RETRY_DELAY = 2.0       # Initial delay between retries in seconds
RETRY_BACKOFF_MULTIPLIER = 1.5  # Multiplier for exponential backoff delay

# Public variables
# Global drone systems
drone = System()            # MAVSDK drone system
currentNedPosition = None   # Holds current NED position for relative movements
offboardActive = False      # Indicates when drone is in motion

# Private Variables
_conversionCache = {}   # Cache variable holds previous conversion results

def get_command_and_values(commandText):
    """
    Extract command type, distance value, and distance type from a given command.

    INPUT: commandText - [string] spoken command

    OUTPUT: [(cmdType, distVal, distType), ...] - list of tuples of parsed commands
    """

    # Normalize commandText to lowercase before processing
    commandLower = commandText.lower().strip()

    # Loop through COMMAND_LOOKUP to find matching trigger phrases
    for trigger, cmdType in COMMAND_LOOKUP.items():
        if commandLower.startswith(trigger):
            remText = commandLower[len(trigger):].strip()
            
            # Check if distance is specified. If one is specified, it is extracted.
            # Otherwise, just the command type is returned
            if not remText:
                return cmdType, None, ""
            
            # Extract numeric value and unit type from remaining text
            distVal = None
            digitMatch = DIGIT_PATTERN.search(remText)

            # If a direct match is found, extract the numeric value
            if digitMatch:
                distVal = int(digitMatch.group())
            # If no direct match is found, attempt to extract spoken numbers
            else:
                total = 0
                for word in remText.split():
                    if word in NUM_DICT:
                        total = total * 100 if NUM_DICT[word] == 100 else total + NUM_DICT[word]
                    elif total > 0:
                        break
                distVal = total if total > 0 else None

            # Determine distance type based on keywords in remaining text
            if cmdType in ("RO_LEFT", "RO_RIGHT"):
                distType = "degrees"
            else:
                distType = "meters"  # Default
                for unitType, pattern in UNIT_PATTERNS.items():
                    if pattern.search(remText):
                        distType = unitType
                        break

            return cmdType, distVal, distType

    # If no trigger is found, return None values
    return None, None, ""

def parse_multiple_commands(commandText):
    """
    Parse multiple commands from a single input string by splitting on separators

    INPUT: commandText - [string] spoken commands

    OUTPUT: commands - [string] list of tuples of parsed commands
    """
    # Split command text into segments based on separators
    commandSegments = SEPARATOR_PATTERN.split(commandText.lower().strip())
    commandSegments = [seg.strip() for seg in commandSegments if seg.strip()]
    
    # If no separators found, treat as single command
    if len(commandSegments) <= 1:
        commandSegments = [commandText.lower().strip()]
    
    # Process segments with early termination for invalid commands
    commands = []
    for segment in commandSegments:
        cmdType, distVal, distType = get_command_and_values(segment)
        if cmdType is not None:
            commands.append((cmdType, distVal, distType))
    
    return commands

async def execute_command(cmdType, distVal, distType):
    """
    Asynchronously executes a command based on the command type, distance value, and distance type

    INPUT: cmdType - [string] command type listed in COMMAND_DICT
           distVal - [int] distance value or None if not specified
           distType - [string] distance type or empty string if not specified
    """
    # Access offboard mode status global variable
    global offboardActive

    # Attempt to execute the command with proper error handling
    try:
        # Convert distances to meters for consistency in calculations
        distanceMeters = convert_to_meters(distVal, distType) if distVal else None

        # In simulation mode, commands are printed to console without actual drone execution
        if SIM_MODE:
            # Commands that stop movement disable offboard mode
            if cmdType in ("DISARM", "STANDBY", "LAND", "STOP"):
                offboardActive = False

            # Print movement commands with distance information if specified
            if cmdType in ("UP", "DOWN", "LEFT", "RIGHT", "FORWARD", "BACKWARD"):
                distStr = f"({distVal} {distType})" if distanceMeters else ""
                print(f"Command: {cmdType}{distStr} [SIMULATION]")
            # Print rotation commands with angle information
            elif cmdType in ("RO_LEFT", "RO_RIGHT"):
                angleDeg = distVal if distVal else 90  # Default rotation angle is 90 degrees
                print(f"Command: {cmdType}({angleDeg} degrees) [SIMULATION]")
            # Print takeoff commands with altitude information if specified
            elif cmdType == "TAKEOFF":
                distStr = f"({distVal} {distType})" if distanceMeters else ""
                print(f"Command: TAKEOFF{distStr} [SIMULATION]")
            # Print all other commands without additional parameters
            else:
                print(f"Command: {cmdType} [SIMULATION]")
            return

        # When not in simulation mode, commands are executed on the actual drone
        # Disable offboard mode when drone stops movement with timeout protection
        if offboardActive and cmdType in ("DISARM", "STANDBY", "LAND", "STOP"):
            try:
                await asyncio.wait_for(drone.offboard.stop(), timeout=OFFBOARD_TIMEOUT)
                offboardActive = False
            except asyncio.TimeoutError:
                print(f"Warning: Timeout stopping offboard mode for {cmdType}")
                offboardActive = False  # Assume it stopped even if we couldn't confirm
            
        # Execute basic drone action commands with timeout protection
        if cmdType == "ARM":
            print("Command: ARM")
            await asyncio.wait_for(drone.action.arm(), timeout=ACTION_TIMEOUT)
        elif cmdType == "DISARM":
            print("Command: DISARM")
            await asyncio.wait_for(drone.action.disarm(), timeout=ACTION_TIMEOUT)
        elif cmdType == "STANDBY":
            print("Command: STANDBY")
            await asyncio.wait_for(drone.action.hold(), timeout=ACTION_TIMEOUT)
        elif cmdType == "LAND":
            print("Command: LAND")
            await asyncio.wait_for(drone.action.land(), timeout=ACTION_TIMEOUT)
        elif cmdType == "STOP":
            print("Command: STOP")
            await asyncio.wait_for(drone.action.hold(), timeout=ACTION_TIMEOUT)
        # Execute takeoff command with optional altitude setting and timeout protection
        elif cmdType == "TAKEOFF":
            if distanceMeters:
                print(f"Command: TAKEOFF({distVal} {distType})")
                await asyncio.wait_for(drone.action.set_takeoff_altitude(distanceMeters), timeout=ACTION_TIMEOUT)
            else:
                print("Command: TAKEOFF")
            await asyncio.wait_for(drone.action.takeoff(), timeout=ACTION_TIMEOUT)
        # Execute movement commands with distance specification and timeout protection
        elif cmdType in ("UP", "DOWN", "LEFT", "RIGHT", "FORWARD", "BACKWARD"):
            if distanceMeters:
                print(f"Command: {cmdType}({distVal} {distType})")
                await asyncio.wait_for(execute_movement_command(cmdType, distanceMeters), timeout=MOVEMENT_TIMEOUT)
            else:
                print(f"Command: {cmdType}")
                # Default movement distance is 1.0 meter when not specified
                await asyncio.wait_for(execute_movement_command(cmdType, 1.0), timeout=MOVEMENT_TIMEOUT)
        # Execute rotation commands with angle specification and timeout protection
        elif cmdType in ("RO_LEFT", "RO_RIGHT"):
            angleDeg = distVal if distVal else 90  # Default rotation angle is 90 degrees
            print(f"Command: {cmdType}({angleDeg} degrees)")
            await asyncio.wait_for(execute_rotation_command(cmdType, angleDeg), timeout=MOVEMENT_TIMEOUT)
            
    # Catch timeout errors specifically to provide clear feedback
    except asyncio.TimeoutError:
        print(f"Timeout error: {cmdType} command took too long to execute")
    # Catch and handle specific MAVSDK exceptions to prevent program crashes
    except ActionError as e:
        print(f"Action error executing {cmdType}: {e}")
    except OffboardError as e:
        print(f"Offboard error executing {cmdType}: {e}")
    except Exception as e:
        print(f"Error executing {cmdType}: {e}")

def convert_to_meters(value, unitType):
    """
    Converts distance values to meters

    INPUT: value - [float] distance value to convert
           unitType - [string] unit type of the input value

    OUTPUT: result - [float] distance value in meters/degrees
    """
    if not value:
        return None

    # Check cache first
    cacheKey = (value, unitType)
    cached_result = _conversionCache.get(cacheKey)
    if cached_result is not None:
        return cached_result

    # Convert value to meters
    if unitType == "feet":
        result = value * 0.3048
    elif unitType == "inches":
        result = value * 0.0254
    elif unitType == "yards":
        result = value * 0.9144
    else:   # Degrees or meters
        result = value

    # Cache result for future use. Limited to 100 entries for performance purposes
    if len(_conversionCache) < 100:
        _conversionCache[cacheKey] = result

    return result

async def execute_movement_command(direction, dist):
    """
    Asynchronously executes a movement command in a specified direction with a given distance.

    INPUT: direction - [string] direction of movement (UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD)
           dist - [float] distance to move in meters
    """
    # Access current NED position and offboard mode status global variables
    global currentNedPosition, offboardActive

    # Initialize current NED position if offboard mode is not active
    if not offboardActive:
        currentNedPosition = [0.0, 0.0, 0.0]

    # Get direction offset vector
    baseOffset = BASE_DIRECTION_OFFSETS.get(direction)
    if baseOffset is None:
        print(f"Warning: Unknown direction '{direction}', no movement will occur")
        return
        
    # Calculate new position
    newNorth = currentNedPosition[0] + (baseOffset[0] * dist)
    newEast = currentNedPosition[1] + (baseOffset[1] * dist)
    newDown = currentNedPosition[2] + (baseOffset[2] * dist)

    # Initialize offboard mode if needed
    if not offboardActive:
        # Set initial position to current location before starting offboard mode
        await asyncio.wait_for(
            drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0)),
            timeout=OFFBOARD_TIMEOUT
        )
        await asyncio.wait_for(drone.offboard.start(), timeout=OFFBOARD_TIMEOUT)
        offboardActive = True

        # Small delay to ensure offboard mode is established
        await asyncio.sleep(0.5)

    # Send movement command
    await asyncio.wait_for(
        drone.offboard.set_position_ned(PositionNedYaw(newNorth, newEast, newDown, 0.0)),
        timeout=OFFBOARD_TIMEOUT
    )

    # Update position tracking
    currentNedPosition[:] = [newNorth, newEast, newDown]
    await asyncio.sleep(2.0)

async def execute_rotation_command(direction, angleDeg):
    """
    Asynchronously executes a rotation command in a specified direction with a given angle.

    INPUT: direction - [string] direction of rotation (RO_LEFT, RO_RIGHT)
           angleDeg - [float] angle to rotate in degrees
    """
    # Access current NED position and offboard mode status global variables
    global currentNedPosition, offboardActive

    # Initialize current NED position if offboard mode is not active
    if not offboardActive:
        currentNedPosition = [0.0, 0.0, 0.0]  # Initialize relative NED position [North, East, Down]

    # Convert angle from degrees to radians for MAVSDK compatibility
    angleRad = math.radians(angleDeg)

    # Left rotation uses negative angle values
    if direction == "RO_LEFT":
        angleRad = -angleRad

    # Initialize offboard mode if not already active with timeout protection
    if not offboardActive:
        await asyncio.wait_for(
            drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0)),
            timeout=OFFBOARD_TIMEOUT
        )
        await asyncio.wait_for(drone.offboard.start(), timeout=OFFBOARD_TIMEOUT)
        offboardActive = True

        # Small delay to ensure offboard mode is established
        await asyncio.sleep(0.5)

    # Send rotation command using yaw angle while maintaining current NED position with timeout protection
    await asyncio.wait_for(
        drone.offboard.set_position_ned(PositionNedYaw(
            currentNedPosition[0], currentNedPosition[1], currentNedPosition[2], angleRad
        )),
        timeout=OFFBOARD_TIMEOUT
    )

    # Wait for the drone to complete the rotation
    await asyncio.sleep(2.0)

async def connect_to_drone():
    """
    Establishes connection to the drone with retry logic and initializes the system for command execution.
    """
    # In simulation mode, skip actual drone connection and display status message
    if SIM_MODE:
        print("SIMULATION MODE")
        print("Voice commands are processed but no drone actions will occur")
        return
    
    # Initialize retry variables for connection attempts with exponential backoff
    retryCount = 0
    retryDelay = INITIAL_RETRY_DELAY

    # Connection retry loop continues until successful connection or max retries exceeded
    while retryCount <= MAX_CONNECTION_RETRIES:
        try:
            # Display connection attempt information to user
            if retryCount == 0:
                print("Connecting to drone...")
            else:
                print(f"Connection attempt {retryCount + 1} of {MAX_CONNECTION_RETRIES + 1}...")

            # Establish UDP connection to drone on standard MAVSDK port with timeout protection
            '''
            TODO: Update connection address for Jetson Nano to connect to Pixhawk 6C:
            Serial Connection - system_address="serial:///dev/ttyUSB0:57600"
            MAVLink over UDP - system_address="udp://:14540"
            Direct Serial Connection - system_address="serial:///dev/ttyACM0:115200"
            '''
            await asyncio.wait_for(
                drone.connect(system_address="udp://:14540"),
                timeout=CONNECTION_TIMEOUT
            )
            
            # Wait for successful connection state before proceeding with timeout protection
            print("Waiting for drone to connect...")
            
            # Define inner function to wait for connection state from drone telemetry
            async def wait_for_connection():
                connectionCheckCount = 0
                maxConnectionChecks = 10  # Limit connection state checks to prevent infinite loops

                async for state in drone.core.connection_state():
                    connectionCheckCount += 1
                    if state.is_connected:
                        print("Drone connected!")
                        return True
                    
                    # Break out if too many checks occur without connection
                    if connectionCheckCount >= maxConnectionChecks:
                        print("Connection state check limit reached")
                        break
                return False
            
            # Attempt to establish connection state within timeout period
            connectionEstablished = await asyncio.wait_for(
                wait_for_connection(),
                timeout=CONNECTION_TIMEOUT
            )
            
            # Verify connection stability after initial connection is established
            if connectionEstablished:
                print("Verifying connection stability...")
                await asyncio.sleep(1.0)    # Brief pause to allow connection to stabilize
                
                # Check if connection is still active after stability pause
                async for state in drone.core.connection_state():
                    if state.is_connected:
                        print("Connection verified and stable!")
                        return  # Success - exit the retry loop
                    else:
                        raise ConnectionError("Connection lost during verification")
            else:
                raise ConnectionError("Failed to establish connection state")
                
        # Handle timeout errors with retry logic and exponential backoff
        except asyncio.TimeoutError:
            retryCount += 1
            if retryCount <= MAX_CONNECTION_RETRIES:
                print(f"Connection timed out after {CONNECTION_TIMEOUT} seconds. Retrying in {retryDelay:.1f} seconds...")
                await asyncio.sleep(retryDelay)
                retryDelay *= RETRY_BACKOFF_MULTIPLIER  # Increase delay for next attempt
            else:
                print(f"Failed to connect after {MAX_CONNECTION_RETRIES + 1} attempts. Connection timed out.")
                raise ConnectionError(f"Drone connection failed after {MAX_CONNECTION_RETRIES + 1} attempts")
                
        # Handle connection errors with retry logic and exponential backoff
        except ConnectionError as e:
            retryCount += 1
            if retryCount <= MAX_CONNECTION_RETRIES:
                print(f"Connection error: {e}. Retrying in {retryDelay:.1f} seconds...")
                await asyncio.sleep(retryDelay)
                retryDelay *= RETRY_BACKOFF_MULTIPLIER  # Increase delay for next attempt
            else:
                print(f"Failed to connect after {MAX_CONNECTION_RETRIES + 1} attempts. Last error: {e}")
                raise ConnectionError(f"Drone connection failed after {MAX_CONNECTION_RETRIES + 1} attempts")
                
        # Handle unexpected errors with retry logic and exponential backoff
        except Exception as e:
            retryCount += 1
            if retryCount <= MAX_CONNECTION_RETRIES:
                print(f"Unexpected error during connection: {e}. Retrying in {retryDelay:.1f} seconds...")
                await asyncio.sleep(retryDelay)
                retryDelay *= RETRY_BACKOFF_MULTIPLIER  # Increase delay for next attempt
            else:
                print(f"Failed to connect after {MAX_CONNECTION_RETRIES + 1} attempts. Last error: {e}")
                raise ConnectionError(f"Drone connection failed after {MAX_CONNECTION_RETRIES + 1} attempts")

# Initialize Vosk speech recognition model for English language processing
model = Model("vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, 16000)

# Initialize PyAudio for real-time audio input capture
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=4096) # Adjust buffer size for optimal performance here and below
stream.start_stream()

print("Audio input stream started. Now listening...")

async def listen():
    """
    Main listening loop that captures audio input and processes voice commands continuously.
    """
    # Access global variables for proper cleanup
    global offboardActive, stream, p

    lastProcessTime = time.time()   # Timestamp of the last audio processing cycle
    minProcessInterval = 0.005      # Minimum time between audio processing cycles

    # Establish drone connection, then begin audio processing loop
    try:
        await connect_to_drone()        
        while True:
            currentTime = time.time()

            # Throttle processing to prevent overwhelming the system
            if currentTime - lastProcessTime < minProcessInterval:
                await asyncio.sleep(0.001)
                continue
            
            try:
                # Non-blocking audio read with smaller buffer
                data = stream.read(4096, exception_on_overflow=False)
                
                # Process audio data through speech recognizer
                if rec.AcceptWaveform(data):
                    result = rec.Result()
                    userCommand = json.loads(result).get("text", "")
                    
                    if userCommand:
                        # Parse and execute commands
                        parsedCommands = parse_multiple_commands(userCommand)

                        if parsedCommands:
                            if len(parsedCommands) > 1:
                                print(f"Executing {len(parsedCommands)} sequential commands:")
                                for i, command_data in enumerate(parsedCommands, 1):
                                    print(f"  Step {i}: ", end="")
                                    await execute_command(*command_data)
                            else:
                                await execute_command(*parsedCommands[0])
                        else:
                            print(f"No valid commands found in: '{userCommand}'")

                    # Update last processing time
                    lastProcessTime = currentTime
            # Catch audio processing errors
            except Exception as audioError:
                print(f"Audio processing error: {audioError}")
                await asyncio.sleep(0.05)
    # Handle user interruption and perform cleanup
    except KeyboardInterrupt:
        print("\nShutting down...")
        await cleanup_resources()
    # Handle general exceptions with cleanup
    except Exception as e:
        print(f"Error in main loop: {e}")
        await cleanup_resources()

async def cleanup_resources():
    """
    Properly clean up all resources including drone offboard mode and audio stream.
    """
    global offboardActive, currentNedPosition, stream, p
    
    try:
        # Stop offboard mode if currently active with timeout protection and resets NED positioning after
        if offboardActive and not SIM_MODE:
            try:
                await asyncio.wait_for(drone.offboard.stop(), timeout=OFFBOARD_TIMEOUT)
            except asyncio.TimeoutError:
                print("Warning: Timeout stopping offboard mode during cleanup")
            finally:
                offboardActive = False
                currentNedPosition = None
    except Exception as e:
        print(f"Error stopping offboard mode: {e}")
    
    try:
        # Clean up audio resources
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()
    except Exception as e:
        print(f"Error cleaning up audio resources: {e}")

# Main program entry point with exception handling
if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")