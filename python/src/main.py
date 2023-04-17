import multiprocessing as mp, numpy as np, pyaudio
import modules.utilities as u, modules.connection as c
from modules.audioprocessing import LFAudioInputHandler, HFAudioInputHandler
from queue import Empty, Full
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio

def audio_producer(control_event, lf_queue: mp.Queue, hf_queue:mp.Queue, chunk_size: int, sample_format, channels: int, sample_rate: int, hf_number_of_chunks: int, exception_on_overflow: bool):
    """
    Opens an audio stream and reads data from it, providing input to the LF and HF processes via multiprocessing Queues.

    Parameters:
    - control_event: checks if the application needs to stop
    - lf_queue: mp.Queue for low level features processes
    - hf_queue: mp.Queue for high level features processes
    - others: various audio parameters
    """
    debugger = u.Debugger()
    debugger.print_success("Audio Producer: Running", flush=True)
    
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sample_rate,
                    frames_per_buffer=chunk_size,
                    input=True)
    except Exception as e:
        debugger.print_error("Couldn't open audio stream")
        u.stop_processes(lf_queue, hf_queue)
        return

    counter = 0
    data_array = []

    while True:
        if control_event.is_set():
            u.stop_processes(lf_queue, hf_queue)
            return

        data = stream.read(chunk_size, exception_on_overflow=exception_on_overflow)
        data_array.append(data)

        try:
            lf_queue.put_nowait(data)
        except Full:
            debugger.print_warning("LF-Queue is full") # If the queue is too big it means that the program takes too much time to process data
            continue

        if counter >= hf_number_of_chunks: # When we have enough data for HLF processing...
            try:
                hf_queue.put_nowait(data_array.copy())
                data_array = []
                counter = 0
                continue
            except Full:
                debugger.print_warning("HF-Queue is full")
                continue

        counter += 1


def lf_audio_consumer(lf_queue: mp.Queue, settings_queue: mp.Queue, chunk_size: int, np_sample_format, channels: int, sample_rate: int):
    """
    Processes Low Level Features from a multiprocessing Queue.

    Parameters:
    - lf_queue: mp.Queue where small chunks of audio data are put by the "audio_producer"
    - others: various audio parameters
    """
    debugger = u.Debugger()
    debugger.print_success("LF Audio Consumer: Running", flush=True)

    lf_audio_input_handlers = []
    for i in range(channels):
        lf_audio_input_handlers.append(LFAudioInputHandler(i, sample_rate, chunk_size, np_sample_format, u.Instruments.VOICE))

    while True:
        data = lf_queue.get()

        try:
            settings = settings_queue.get_nowait()
            lf_audio_input_handlers[settings[0]].set_instrument(settings[1])
            debugger.print_info("Channel ("+str(settings[0])+") was set to "+settings[1].get_string())
        except Empty:
            pass

        if data is None: break
        data = np.frombuffer(data, np_sample_format)

        data_channels = []
        for i in range(channels):
            data_channels.append(data[i::channels])

        try: 
            for i in range(channels):
                lf_audio_input_handlers[i].process(data_channels[i])
        except Exception as e:
            debugger.print_error("LF - Error while processing input")
            raise e
            print(e)
            return


def hf_audio_consumer(hf_queue: mp.Queue, chunk_size: int, np_sample_format, sample_rate: int):
    """
    Processes High Level Features from a multiprocessing Queue.

    Parameters:
    - hf_queue: mp.Queue where medium chunks (some seconds) of audio data are put by the "audio_producer"
    - others: various audio parameters
    """
    debugger = u.Debugger()
    debugger.print_success("HF Audio Consumer: Running", flush=True)

    hf_audio_input_handler = HFAudioInputHandler(0, sample_rate, chunk_size, np_sample_format)

    while True:
        data = hf_queue.get()
        if data is None: break
        try:
            hf_audio_input_handler.process(data)
        except Exception as e:
            debugger.print_error("HF - Error while processing input")
            print(e)
            return

async def init_main():
    debugger = u.Debugger()
    u.EXTERNAL_OSC_CONTROLLER = u.wait_for_start_input()  

    debugger.print_info("Setting Up Variables...")      
    lf_queue = mp.Queue() # Queues Initialization
    hf_queue = mp.Queue()
    settings_queues = []

    for _ in range(u.NUMBER_OF_LF_PROCESSES):
        settings_queues.append(mp.Queue())
    
    if u.EXTERNAL_OSC_CONTROLLER:
        debugger.print_info("Setting Up Server...")
        dispatcher = c.create_dispatcher(settings_queues)
        server = AsyncIOOSCUDPServer((u.IN_NET_ADDRESS, u.IN_NET_PORT), dispatcher, asyncio.get_event_loop())
        transport, protocol = await server.create_serve_endpoint()

    await main(lf_queue, hf_queue, settings_queues)
    if u.EXTERNAL_OSC_CONTROLLER: transport.close()

async def main(lf_queue, hf_queue, settings_queues):
    debugger = u.Debugger()

    if u.EXTERNAL_OSC_CONTROLLER:
        print("Waiting for connection message from OSC Controller", end="")
        n_dots = 0
        while not u.EXTERNAL_OSC_CONTROLLER_CONNECTED:
            if n_dots == 3:
                print(end='\b\b\b', flush=True)
                print(end='   ',    flush=True)
                print(end='\b\b\b', flush=True)
                n_dots = 0
            else:
                print(end='.', flush=True)
                n_dots += 1
            await asyncio.sleep(.5)

    debugger.print_info("Setting up processes...")
    event = mp.Event()

    audio_reader = mp.Process(target=audio_producer, args=(
        event,
        lf_queue,
        hf_queue,
        u.CHUNK_SIZE,
        u.SAMPLE_FORMAT,
        u.CHANNELS,
        u.SAMPLE_RATE,
        u.HF_NUMBER_OF_CHUNKS,
        u.EXCEPTION_ON_OVERFLOW,
    ))

    # A process is created for each user-specified number of cores that are going to extract low-level features (default = 2)
    lf_consumers = []
    for i in range(u.NUMBER_OF_LF_PROCESSES):
        lf_consumers.append(mp.Process(target=lf_audio_consumer, args=(
            lf_queue, 
            settings_queues[i],
            u.CHUNK_SIZE, 
            u.NP_SAMPLE_FORMAT, 
            u.CHANNELS,
            u.SAMPLE_RATE,
        )))

    # A process is created for each user-specified number of cores that are going to extract low-level features (default = 1)
    hf_consumers = []
    for i in range(u.NUMBER_OF_HF_PROCESSES):
        hf_consumers.append(mp.Process(target=hf_audio_consumer, args=(
            hf_queue,
            u.CHUNK_SIZE,
            u.SAMPLE_FORMAT,
            u.SAMPLE_RATE,
        )))

    # Starts all the processes
    for p in hf_consumers:
        p.start()
    for p in lf_consumers:
        p.start()
    audio_reader.start()

    while not u.STOP: await asyncio.sleep(1)
    event.set()

    debugger.print_info("Shutting Down...")
    # Waits for all processes to finish
    for p in hf_consumers:
        p.join()
    for p in lf_consumers:
        p.join()
    audio_reader.join()

# Main protection to ensure correct multiprocessing behaviour
if __name__ == "__main__":
    asyncio.run(init_main())