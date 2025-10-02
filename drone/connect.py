'''
connect.py
Handles drone connection and connection termination via MAVSDK.
'''
import asyncio
from mavsdk import System
from config import (
    SIM_MODE,
    DRONE_CONNECTION_PATH,
    CONNECTION_TIMEOUT,
    MAX_CONNECTION_RETRIES,
    OFFBOARD_TIMEOUT,
    INITIAL_RETRY_DELAY,
    RETRY_BACKOFF_MULTIPLIER
)
from .state import (
    get_instance,
    reset_instance,
    get_offboard_state
)

async def wait_for_connection():
    """Attempt to connect to the drone via drone telemetry."""
    # Attempt to confirm connection within the timeout period.
    drone = get_instance()
    startTime = asyncio.get_event_loop().time()

    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Drone connection confirmed.")
            return True

        # Break if timeout is exceeded
        if (asyncio.get_event_loop().time() - startTime) >= CONNECTION_TIMEOUT:
            print("Connection attempt timed out.")
            break

        await asyncio.sleep(0.1)

    return False

async def connect_to_drone():
    """Establish a connection to the drone."""
    retryDelay = INITIAL_RETRY_DELAY    # Current delay between retries

    # Establish the drone instance
    drone = get_instance()

    # In simulation mode, skip the connection process
    if SIM_MODE:
        print("Simulation mode enabled. Skipping drone connection.")
        return None
    
    # Attempt to connect to the drone while under the retry limit
    retryCount = 0
    while retryCount < MAX_CONNECTION_RETRIES:
        try:
            # Displays connection attempt message
            print(f"Attempting to connect to drone (Attempt {retryCount + 1}/{MAX_CONNECTION_RETRIES})...")

            await asyncio.wait_for(
                drone.connect(system_address=DRONE_CONNECTION_PATH),
                timeout=CONNECTION_TIMEOUT
            )

            # Attempt to establish connection within the timeout period
            isConnectionEstablished = await asyncio.wait_for(
                wait_for_connection(),
                timeout=CONNECTION_TIMEOUT
            )

            # Verify connection stability
            if isConnectionEstablished:
                print("Successfully connected to the drone. Verifying stability...")
                await asyncio.sleep(1.0)

                try:
                    isConnectionStable = await asyncio.wait_for(
                        wait_for_connection(),
                        timeout=CONNECTION_TIMEOUT
                    )
                    if isConnectionStable:
                        print("Connection stable.")
                        return
                    else:
                        raise ConnectionError("Connection lost during verification.")
                except asyncio.TimeoutError:
                    raise ConnectionError("Connection verification timed out.")
            else:
                raise ConnectionError("Failed to establish connection.")
        
        # Timeout error handling
        except asyncio.TimeoutError:
            retryCount += 1
            if retryCount < MAX_CONNECTION_RETRIES:
                print(f"Connection attempt timed out. Retrying in {retryDelay} seconds...")
                await asyncio.sleep(retryDelay)
                retryDelay *= RETRY_BACKOFF_MULTIPLIER
            else:
                print("Maximum connection attempts reached. Unable to connect to the drone.")
                raise ConnectionError(f"Drone connection failed after {MAX_CONNECTION_RETRIES} attempts.")
        
        # Connection error handling
        except ConnectionError as e:
            retryCount += 1
            if retryCount < MAX_CONNECTION_RETRIES:
                print(f"Connection error occurred: {e}. Retrying in {retryDelay} seconds...")
                await asyncio.sleep(retryDelay)
                retryDelay *= RETRY_BACKOFF_MULTIPLIER
            else:
                print(f"Maximum connection attempts reached. Unable to connect to the drone. Last error: {e}")
                raise ConnectionError(f"Drone connection failed after {MAX_CONNECTION_RETRIES} attempts.")
        
        # Unexpected error handling
        except Exception as e:
            retryCount += 1
            if retryCount < MAX_CONNECTION_RETRIES:
                print(f"Unexpected error occurred: {e}. Retrying in {retryDelay} seconds...")
                await asyncio.sleep(retryDelay)
                retryDelay *= RETRY_BACKOFF_MULTIPLIER
            else:
                print(f"Maximum connection attempts reached. Unable to connect to the drone. Last error: {e}")
                raise ConnectionError(f"Drone connection failed after {MAX_CONNECTION_RETRIES} attempts.")

async def cleanup_drone():
    """Cleans drone resources on shutdown."""
    # Clean offboard mode and resets NED position
    try:
        if get_offboard_state() and not SIM_MODE:
            drone = get_instance()
            try:
                await asyncio.wait_for(drone.offboard.stop(), timeout=OFFBOARD_TIMEOUT)
            except asyncio.TimeoutError:
                print("Timeout while stopping offboard mode during cleanup.")
            finally:
                reset_instance()
    
    # General exception handling
    except Exception as e:
        print(f"Error disabling offboard mode: {e}")