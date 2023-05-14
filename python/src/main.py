import modules.default_parameters as dp
import os
from multiprocessing import Process, Queue, Event
from queue import Full, Empty
from modules.setup import SetupHandler
from modules.audioprocessing import LFAudioInputHandler, HFAudioInputHandler
from modules.audio_producer import AudioProducer
from modules.utilities import *
from modules.custom_exceptions import *


def stop_execution(lf_queue: Queue, hf_queue: Queue, streams: list):
    for _ in range(dp.CPU_PARAMETERS['numLfCores']):
        lf_queue.put(None)
    for _ in range(dp.CPU_PARAMETERS['numHfCores']):
        hf_queue.put(None)
    for s in streams:
        if s:
            s.stop_stream()
            s.close()


def audio_producer(audio_producer_object: AudioProducer, control_event, lf_queue: Queue, hf_queue: Queue, parameters: dict):
    sh = SetupHandler.get_instance()
    sh.set_audio_parameters(parameters)
    in_stream, out_stream = sh.get_audio_streams()

    print_success("Started audio producer process")
    while True:
        try:
            if control_event.is_set():
                stop_execution(lf_queue, hf_queue, [in_stream, out_stream])
                return

            audio_chunk = audio_producer_object.get_next_chunk(in_stream, out_stream)

            lf_queue.put_nowait(audio_chunk)
            hf_queue.put_nowait(audio_chunk)
        except Full:
            print_warning("Queue is full")
            continue
        except FinishedSongException as e:
            print_info("Song is finished")
            stop_execution(lf_queue, hf_queue, [in_stream, out_stream])
            return
        except AudioProducingException as e:
            print_error(e)
        except Exception as e:
            print_error("Something bad happened while producing audio")
            print_dbg(e)


def lf_audio_consumer(lf_queue: Queue, settings_queue: Queue, parameters: dict):
    lf_audio_input_handlers = []
    channels = parameters['channels']
    instruments = parameters['instruments']
    for i in range(channels):
        lf_audio_input_handlers.append(LFAudioInputHandler(parameters, i, instruments[i]))
    
    print_success("Started LF consumer process")
    while True:
        data = lf_queue.get()
        if data is None:
            print_info("Shutting down LF process...")
            break

        try:
            channel, settings = settings_queue.get_nowait()
            for handler in lf_audio_input_handlers:
                if handler.channel == channel:
                    handler.handle_settings(settings)
                    break
        except Empty:
            pass

        try:
            for i in range(len(lf_audio_input_handlers)):
                lf_audio_input_handlers[i].process(data[i])
        except Exception as e:
            print_error("Something bad happened while processing audio (LLF)")
            print_dbg(e)


def hf_audio_consumer(hf_queue: Queue, parameters: dict):
    hf_audio_input_handler = HFAudioInputHandler(parameters, 0, Instruments.DEFAULT)

    print_success("Started HF consumer process")
    while True:
        data = hf_queue.get()
        if data is None:
            print_info("Shutting down HF process...")
            break
        try:
            hf_audio_input_handler.process(data)
        except Exception as e:
            print_error("Something bad happened while processing audio (HLF)")
            print_dbg(e)


if __name__ == "__main__":
    sh = SetupHandler.get_instance()
    sh.set_main_path(os.path.dirname(__file__))
    parameters = sh.setup()
    audio_producer_object = sh.get_audio_producer()

    control_event = Event()

    cpu_parameters = dp.CPU_PARAMETERS
    lf_queue = Queue()
    hf_queue = Queue()
    settings_queue = Queue()

    processes = []
    for _ in range(cpu_parameters['numLfCores']):
        processes.append(Process(target=lf_audio_consumer, args=(
            lf_queue,
            settings_queue,
            parameters,
        )))
    for _ in range(cpu_parameters['numHfCores']):
        processes.append(Process(target=hf_audio_consumer, args=(
            hf_queue,
            parameters,
        )))

    processes.append(Process(target=audio_producer, args=(
        audio_producer_object,
        control_event,
        lf_queue,
        hf_queue,
        parameters,
    )))

    for p in processes:
        p.start()

    for p in processes:
        p.join()
