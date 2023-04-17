import pyaudio, numpy as np
from enum import Enum

"""
Classes declarations
"""
class Normalizations(Enum):
    NONE = 0
    PEAK = 1
    RMS = 2
    Z_SCORE = 3
    MIN_MAX = 4

class Instruments(Enum):
    DEFAULT = 1
    VOICE = 2
    GUITAR = 3
    PIANO = 4
    STRINGS = 5
    DRUMS = 6

    def get_string(self):
        """
        Returns the name of the instrument as a string
        """
        return self.name
    
    @staticmethod
    def from_string(s: str):
        if s == "DEFAULT": return Instruments.DEFAULT
        if s == "VOICE": return Instruments.VOICE
        if s == "GUITAR": return Instruments.GUITAR
        if s == "PIANO": return Instruments.PIANO
        if s == "STRINGS": return Instruments.STRINGS
        if s == "DRUMS": return Instruments.DRUMS
        raise Exception("Couldn't parse string into instrument")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Debugger:
    is_active = True

    def __init__(self):
        pass

    def print_success(self, string, flush=False):
        if Debugger.is_active:
            print(bcolors.OKGREEN+"[OK] "+str(string)+bcolors.ENDC, flush=flush)

    def print_info(self, string, flush=False):
        if Debugger.is_active:
            print(bcolors.OKBLUE+"[INFO] "+bcolors.UNDERLINE+str(string)+bcolors.ENDC, flush=flush)

    def print_data(self, string, flush=False):
        if Debugger.is_active:
            print(bcolors.OKCYAN+"[DATA] "+str(string)+bcolors.ENDC, flush=flush)

    def print_warning(self, string, flush=False):
        if Debugger.is_active:
            print(bcolors.WARNING+bcolors.BOLD+"[WARNING] "+str(string)+bcolors.ENDC, flush=flush) 

    def print_error(self, string, flush=False):
        if Debugger.is_active:
            print(bcolors.FAIL+bcolors.BOLD+"[ERROR] "+str(string)+bcolors.ENDC, flush=flush) 

    def dbg(self, string, flush=False):
        if Debugger.is_active:
            print(bcolors.OKGREEN+"[DBG] "+str(string)+bcolors.ENDC, flush=flush)

"""
Functions declarations
"""

## Not Audio Related
def wait_for_start_input():
    yes = ["yes", "y"]
    no = ["no", "n"]

    while True:    
        user_input = input("Using External OSC Controller? (y/n): ")
        if user_input in yes: return True
        if user_input in no: return False

def stop_processes(lf_queue, hf_queue):
    for _ in range(NUMBER_OF_LF_PROCESSES):
        lf_queue.put(None)
    for _ in range(NUMBER_OF_HF_PROCESSES):
        hf_queue.put(None)

## Audio Related
def get_default_max_number_of_tracks():
    channels = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels']
    return int(channels)

def get_default_sample_rate():
    sr = pyaudio.PyAudio().get_default_input_device_info()['defaultSampleRate']
    return int(sr)

def get_fundamental_frequency_range_by_instrument(inst: Instruments):
    if inst == Instruments.DEFAULT:
        return [20.0, 8000.0]
    if inst == Instruments.VOICE:
        return [80.0, 4000.0]
    if inst == Instruments.GUITAR:
        return [20.0, 5000.0]
    if inst == Instruments.PIANO:
        return [20.0, 4500.0]
    if inst == Instruments.STRINGS:
        return [20.0, 3500.0]
    if inst == Instruments.DRUMS:
        return None

"""
Ummutable constants (defined on startup)
"""
## Net and communication
NET_ADDRESS = "127.0.0.1"
NET_PORT = 12345

IN_NET_ADDRESS = "127.0.0.1"
IN_NET_PORT = 1337

## Outgoing addresses
OSC_LF_ADDRESS = "/LFmsg_ch"
OSC_HF_ADDRESS = "/HFmsg_ch"

# Incoming addresses
EXTERNAL_OSC_CONTROLLER = None
EXTERNAL_OSC_CONTROLLER_CONNECTED = False
OSC_START_ADDRESS = "/START"
OSC_STOP_ADDRESS = "/STOP"
OSC_CHANNEL_SETTINGS_ADDRESS = "/ch_settings"
STOP = False

## PyAudio
CHUNK_SIZE = 1024 * 4 # number of frames per buffer
SAMPLE_FORMAT = pyaudio.paFloat32 # format of read sample
CHANNELS = 2 # sound card channels
if get_default_max_number_of_tracks() < CHANNELS:
    CHANNELS = get_default_max_number_of_tracks()

SAMPLE_RATE = 44100 # sample rate in Hz
if get_default_sample_rate() != SAMPLE_RATE:
    SAMPLE_RATE = get_default_sample_rate()


EXCEPTION_ON_OVERFLOW = False


## Audio Processing
CHANNEL_AUDIO_THRESHOLD = 0.005
DEFAULT_NFFT = 4096
HOP_LENGTH = int(DEFAULT_NFFT/2)
WINDOW_SIZE = 4096
WINDOW_TYPE = 'hann'
NP_SAMPLE_FORMAT = np.float32
HF_NUMBER_OF_CHUNKS = 200
PITCH_FMIN = 50
PITCH_FMAX = 6000
PITCH_THRESHOLD = 0.2

## CPU
NUMBER_OF_LF_PROCESSES = 2
NUMBER_OF_HF_PROCESSES = 1

NORM_TYPE = Normalizations.PEAK