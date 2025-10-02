'''
main.py
Main execution point for Voice Command Recognition and MAVLink Communication.
Script contains audio initialization and main listening loop.
Please refer to README.md for setup and usage instructions before running this script.
'''
import asyncio
import time
from config import SIM_MODE, PROCESS_INTERVAL, AUDIO_BUFFER_SIZE
from voice.parser import parse_commands
from drone.command import execute_command
from drone.connect import connect_to_drone, cleanup_drone
from utils import (
    get_stop_requested,
    set_stop_requested,
    get_command_executing,
    set_command_executing
)
from voice.recognizer import (
    initialize_speech_recognition,
    initialize_audio_stream,
    process_audio,
    cleanup_audio
)

# Global variables
model = None    # Vosk model
rec = None      # Kaldi recognizer
p = None        # PyAudio instance
stream = None   # Audio stream

def initialize_audio():
    """Initialize speech recognition and audio stream for audio capture."""
    global model, rec, p, stream

    try:
        # Initialize Vosk model
        model, rec = initialize_speech_recognition()
        print("Vosk model loaded.")

        # Initialize PyAudio for audio capture
        p, stream = initialize_audio_stream()
        print("Audio stream started. Now listening for commands:")
    
    # General exception handling
    except Exception as e:
        print(f"Error during initialization: {e}")
        cleanup_audio(p, stream)
        raise

async def process_voice_commands():
    """Separate coroutine for processing voice commands."""
    global stream, rec
    
    prevProcessTime = time.time()   # Last time audio was processed
    
    while True:
        try:
            currentTime = time.time()   # Time at start of loop iteration
            
            # If not enough time has passed, wait
            if currentTime - prevProcessTime < PROCESS_INTERVAL:
                await asyncio.sleep(0.02)
                continue
            
            # Read and process audio with error handling
            try:
                data = stream.read(AUDIO_BUFFER_SIZE, exception_on_overflow=False)  # Captured audio data
                command = process_audio(rec, data)                                  # Recognized command from audio data
            
            # General exception handling 
            except Exception as audioError:
                print(f"Audio read error: {audioError}")
                await asyncio.sleep(0.1)
                continue
            
            if command:
                print(f"Recognized: '{command}'")
                
                # Check for immediate STOP command
                isExecuting = get_command_executing()
                if isExecuting and ('stop' in command.lower()):
                    # Fast path for stop commands
                    parsedCommands = parse_commands(command)
                    if parsedCommands and parsedCommands[0][0] == "STOP":
                        print("STOP command received. Interrupting current operation.")
                        set_stop_requested(True)
                        continue
                
                # Only process new commands if nothing is executing
                if not isExecuting:
                    parsedCommands = parse_commands(command)

                    if parsedCommands:
                        # Create task to execute commands without blocking audio loop
                        asyncio.create_task(execute_command_chain(parsedCommands))
                    else:
                        print(f"No valid commands found in: '{command}'")
            
            prevProcessTime = currentTime

        # General exception handling
        except Exception as e:
            print(f"Audio processing error: {e}")
            await asyncio.sleep(0.1)

async def execute_command_chain(parsedCommands):
    """
    Execute a chain of commands with interrupt support.
    Args:
        parsedCommands: List of parsed command tuples (cmdType, distVal, distType)
    """
    try:
        set_command_executing(True)
        set_stop_requested(False)
        
        if len(parsedCommands) > 1:
            print(f"Executing {len(parsedCommands)} sequential commands:")
            for i, commandData in enumerate(parsedCommands, 1):
                # Check for stop request before each command
                if get_stop_requested():
                    print("Command chain interrupted by STOP command!")
                    break
                    
                print(f"    Step {i}: ", end="")
                await execute_command(*commandData)
        else:
            await execute_command(*parsedCommands[0])
    
    # General exception handling
    except Exception as e:
        print(f"Error executing command chain: {e}")
    finally:
        set_command_executing(False)
        set_stop_requested(False)

async def listen():
    """Main listening loop with concurrent command processing."""
    try:
        await connect_to_drone()
        
        # Start voice processing as a background task
        voice_task = asyncio.create_task(process_voice_commands())
        
        # Keep the main loop running
        try:
            await voice_task
        
        # User interrupt handling
        except KeyboardInterrupt:
            voice_task.cancel()
            raise

    # User interrupt handling        
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
        await cleanup()
    # General exception handling. Clean resources before exiting
    except Exception as e:
        print(f"Unexpected error in listen loop: {e}")
        await cleanup()

async def cleanup():
    """Clean all resources."""
    global p, stream

    try:
        # Clean drone resources
        await cleanup_drone()

        # Clean audio resources
        cleanup_audio(p, stream)
        print("Audio resources cleaned.")

        print("Cleanup complete. Exiting...")
    
    # General exception handling
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Main function
async def main():
    """Main function"""
    try:
        # Initialize audio systems
        initialize_audio()

        # Start listening for commands
        await listen()
    
    # User interrupt handling
    except KeyboardInterrupt:
        print("Keyboard interrupt received in main. Exiting...")
    # General exception handling
    except Exception as e:
        print(f"Unexpected error in main: {e}")
    finally:
        await cleanup()

# Program entry point
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")