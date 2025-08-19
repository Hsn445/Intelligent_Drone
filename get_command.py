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
from vosk import Model, KaldiRecognizer
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed, OffboardError
from mavsdk.telemetry import Position

# Command dictionary maps spoken phrases to command types
commandDict = {
    "ARM":      ["arm drone", "arm"],
    "DISARM":   ["disarm drone", "disarm"],
    "STANDBY":  ["standby", "stand by"],
    "LAND":     ["land", "land drone"],
    "STOP":     ["stop", "stop drone"],
    "TAKEOFF":  ["takeoff", "take off"],
    "UP":       ["up"],
    "DOWN":     ["down"],
    "LEFT":     ["left"],
    "RIGHT":    ["right"],
    "FORWARD":  ["forward"],
    "BACKWARD": ["backward"],
    "RO_LEFT":  ["turn left", "rotate left"],
    "RO_RIGHT": ["turn right", "rotate right"]
}

# Number dictionary converts spoken numbers to integer variables
numDict = {
    'one': 1, 'two': 2, 'to': 2, 'three': 3, 'four': 4, 'for': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100
}

# Separator list contains phrases that identify and separate multiple commands in an input string
separators = [
    ',',
    ', then ',
    ' then ',
    ', and ',
    ' and ',
    ', next ',
    ' next ',
    ' after that ',
    ' followed by ',
    ' afterward '
]

# Global drone systems
drone = System()        # MAVSDK drone system
currentPosition = None  # Holds current drone position
offboardActive = False  # Indicates when drone is in motion
SIM_MODE = True         # Sets simulation mode; should be False when running with Pixhawk

def get_command_and_values(commandText):
    """
    Extract command type, distance value, and distance type from a given command.
    
    INPUT: commandText - [string] spoken command
    
    OUTPUT: cmdType - [string] command type listed in commandDict
            distVal - [int] distance value or None if not specified
            distType - [string] distance type or empty string if not specified
    """

    # commandText normalized to lowercase before processing
    commandLower = commandText.lower()
    
    # Loops through commandDict to find trigger phrases
    for cmdType, triggers in commandDict.items():
        for trigger in triggers:
            if commandLower.startswith(trigger):
                remText = commandLower[len(trigger):].strip()   # Remaining text after command trigger

                # Checks if distance is specified. If one is specified, it is extracted. Otherwise, just the command type is returned
                if not remText:
                    return cmdType, None, ""
                
                # Extract numeric value and unit type
                distVal = None
                digitMatch = re.search(r'\d+', remText) # Parses numeric digits from remaining text
                
                # If digitMatch is found in numDict, extract it
                if digitMatch:
                    distVal = int(digitMatch.group())
                else:
                    # If no direct digitMatch is found, iterate through remText and extract numbers to find distance
                    total = 0   # Distance value accumulator
                    for word in remText.split():
                        if word in numDict:
                            total = total * 100 if numDict[word] == 100 else total + numDict[word]
                        elif total > 0:
                            break
                    distVal = total if total > 0 else None

                # Determine distance type based on remaining text
                if cmdType in ["RO_LEFT", "RO_RIGHT"]:
                    distType = "degrees"
                elif any(unit in remText for unit in ['inch']):
                    distType = "inches"
                elif any(unit in remText for unit in ['feet', 'foot']):
                    distType = "feet"
                elif 'yard' in remText:
                    distType = "yards"
                else:
                    distType = "meters"

                return cmdType, distVal, distType

    # If no command type is matched, return None values
    return None, None, ""

def parse_multiple_commands(commandText):
    """
    Parse multiple commands from a single input.

    INPUT: commandText - [string] spoken command(s)

    OUTPUT: [(cmdType, distVal, distType), ...] - list of tuples of parsed commands
    """
    commandSegments = [commandText.lower().strip()] # Take the entire command as a single segment
    
    # Split command based on defined separators (e.g., commas, 'then', 'and')
    for separator in separators:
        new_segments = []
        for segment in commandSegments:
            new_segments.extend(part.strip() for part in segment.split(separator) if part.strip())
        commandSegments = new_segments
    
    # Process each command segment to extract command type, distance value, and distance type
    return [
        (cmdType, distVal, distType)
        for segment in commandSegments
        for cmdType, distVal, distType in [get_command_and_values(segment)]
        if cmdType is not None
    ]

async def execute_command(cmdType, distVal, distType):
    """
    Asynchronously executes a command based on the command type, distance value, and distance type.

    INPUT: cmdType - [string] command type listed in commandDict
           distVal - [int] distance value or None if not specified
           distType - [string] distance type or empty string if not specified
    """
    # Grabs current drone position and offboard mode status global variables
    global currentPosition, offboardActive

    # Attempts to execute a command
    try:
        distanceMeters = convert_to_meters(distVal, distType) if distVal else None # Distances converted to meters for consistency

        # Commands printed in console while in simulation mode
        if SIM_MODE:
            if cmdType in ["DISARM", "STANDBY", "LAND", "STOP"]:
                offboardActive = False

            if cmdType in ["UP", "DOWN", "LEFT", "RIGHT", "FORWARD", "BACKWARD"]:
                distStr = f"({distVal} {distType})" if distanceMeters else ""
                print(f"Command: {cmdType}{distStr} [SIMULATION]")
            elif cmdType in ["RO_LEFT", "RO_RIGHT"]:
                angleDeg = distVal if distVal else 90  # Rotation angle. Defaults to 90 degrees
                print(f"Command: {cmdType}({angleDeg} degrees) [SIMULATION]")
            elif cmdType == "TAKEOFF":
                distStr = f"({distVal} {distType})" if distanceMeters else ""
                print(f"Command: TAKEOFF{distStr} [SIMULATION]")
            else:
                print(f"Command: {cmdType} [SIMULATION]")
            return

        # When outside of simulation mode, commands are executed on the drone
        # Offboard mode is disabled when drone stops movement
        if offboardActive and cmdType in ["DISARM", "STANDBY", "LAND", "STOP"]:
            await drone.offboard.stop()
            offboardActive = False
        if cmdType == "ARM":
            print("Command: ARM")
            await drone.action.arm()
        elif cmdType == "DISARM":
            print("Command: DISARM")
            await drone.action.disarm()
        elif cmdType == "STANDBY":
            print("Command: STANDBY")
            await drone.action.hold()
        elif cmdType == "LAND":
            print("Command: LAND")
            await drone.action.land()
        elif cmdType == "STOP":
            print("Command: STOP")
            await drone.action.hold()
        # For movement commands, offboard mode is activated and distance is set, if specified
        elif cmdType == "TAKEOFF":
            if distanceMeters:
                print(f"Command: TAKEOFF({distVal} {distType})")
                await drone.action.set_takeoff_altitude(distanceMeters)
            else:
                print("Command: TAKEOFF")
            await drone.action.takeoff()
        elif cmdType in ["UP", "DOWN", "LEFT", "RIGHT", "FORWARD", "BACKWARD"]:
            if distanceMeters:
                print(f"Command: {cmdType}({distVal} {distType})")
                await execute_movement_command(cmdType, distanceMeters)
            else:
                print(f"Command: {cmdType}")
                await execute_movement_command(cmdType, 1.0)
        elif cmdType in ["RO_LEFT", "RO_RIGHT"]:
            angleDeg = distVal if distVal else 90
            print(f"Command: {cmdType}({angleDeg} degrees)")
            await execute_rotation_command(cmdType, angleDeg)
    # The following exceptions are caught and printed to the console:
    # ActionError is raised when an action command fails
    # OffboardError is raised when offboard mode fails to start or stop
    # Exceptions are caught to prevent the program from crashing
    except ActionError as e:
        print(f"Action error executing {cmdType}: {e}")
    except OffboardError as e:
        print(f"Offboard error executing {cmdType}: {e}")
    except Exception as e:
        print(f"Error executing {cmdType}: {e}")

def convert_to_meters(value, unitType):
    """
    Converts a given distance value not in meters to meters.

    INPUT: value - [int] distance value
           unitType - [string] unit type
    
    OUTPUT: [float] distance value in meters or None if value is not specified
    """
    if not value:
        return None
    
    conversion_factors = {
        "meters": 1.0,
        "feet": 0.3048,
        "inches": 0.0254,
        "yards": 0.9144,
        "degrees": 1.0
    }
    
    return value * conversion_factors.get(unitType, 1.0)

async def execute_movement_command(direction, dist):
    """
    Asynchronously executes a movement command in a specified direction with a given distance.

    INPUT: direction - [string] direction of movement
           dist - [float] distance to move in meters
    """
    # Gets current drone position and offboard mode status global variables
    global currentPosition, offboardActive

    # Gets current drone position
    if not currentPosition:
        async for position in drone.telemetry.position():
            currentPosition = position
            break

    # Gets the NED coordinate offsets for the specified direction
    directionOffsets = {
        "FORWARD": (dist, 0, 0),
        "BACKWARD": (-dist, 0, 0),
        "LEFT": (0, -dist, 0),
        "RIGHT": (0, dist, 0),
        "UP": (0, 0, -dist),
        "DOWN": (0, 0, dist)
    }

    northOffset, eastOffset, downOffset = directionOffsets.get(direction, (0, 0, 0))

    # Starts offboard mode if not already active
    if not offboardActive:
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        offboardActive = True

    # Perform the movement command
    await drone.offboard.set_position_ned(PositionNedYaw(
        northOffset, eastOffset, downOffset, 0.0
    ))

    # Delay for the drone to move
    await asyncio.sleep(2.0)

async def execute_rotation_command(direction, angleDeg):
    """
    Asynchronously executes a rotation command in a specified direction with a given angle.

    INPUT: direction - [string] direction of rotation
           angleDeg - [float] angle to rotate in degrees
    """
    # Gets current drone position and offboard mode status global variables
    global currentPosition, offboardActive

    # Converts angle from degrees to radians
    angleRad = math.radians(angleDeg)
    if direction == "RO_LEFT":
        angleRad = -angleRad

    # Starts offboard mode if not already active
    if not offboardActive:
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        offboardActive = True

    # Performs rotation command
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, angleRad))

    # Delay for the drone to rotate
    await asyncio.sleep(2.0)

async def connect_to_drone():
    """
    Connects to the drone and initializes the connection.
    """
    # If drone is in simulation mode, print a message and return
    if SIM_MODE:
        print("SIMULATION MODE")
        print("Voice commands are processed but no drone actions will occur")
        return
    
    print("Connecting to drone...")
    await drone.connect(system_address="udp://:14540")
    
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    # Initialize current position
    global currentPosition
    async for position in drone.telemetry.position():
        currentPosition = position
        break

# Initialize language model and speech recognizer
model = Model("vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, 16000)

# Initialize PyAudio and begin audio input stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=4096)
stream.start_stream()

print("Audio input stream started. Now listening...")

async def listen():
    """
    Listens to audio input and processes commands.
    """
    # Connects to drone. If connection occurs, take in audio input and process commands
    try:
        await connect_to_drone()
        
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                userCommand = json.loads(rec.Result()).get("text", "")
                if not userCommand:
                    continue

                parsed_commands = parse_multiple_commands(userCommand)

                if parsed_commands:
                    if len(parsed_commands) > 1:
                        print(f"Executing {len(parsed_commands)} sequential commands:")
                        for i, command_data in enumerate(parsed_commands, 1):
                            print(f"  Step {i}: ", end="")
                            await execute_command(*command_data)
                    else:
                        await execute_command(*parsed_commands[0])
                else:
                    print(f"No valid commands found in: '{userCommand}'")
    # Exception: User interrupts the program and shuts it down
    except KeyboardInterrupt:
        print("\nShutting down...")
        if offboardActive:
            await drone.offboard.stop()
        stream.stop_stream()
        stream.close()
        p.terminate()
    # Exception: General error handling
    except Exception as e:
        print(f"Error in main loop: {e}")
        if offboardActive:
            await drone.offboard.stop()
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")