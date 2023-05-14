from __future__ import annotations
import os
import zipfile
import glob
import pyaudio
import numpy as np
from scipy.io.wavfile import read
from modules.audio_producer import AudioProducer, LiveAudioProducer, RecordedAudioProducer
from modules.default_parameters import AUDIO_PROCESSING_PARAMETERS
from modules.custom_exceptions import *
from modules.utilities import Instruments

class SetupHandler:
    """Singleton that handles the setup phase of the application and stores audio parameters.
    """
    __instance = None

    def __init__(self):
        """Constructor to not be accessed directly (Singleton pattern).
        It inizializes the :py:class:`FileHandler` to :py:type:`None` and creates an empty
        :py:type:`dict` that will contain the audio parameters.
        """
        if SetupHandler.__instance is not None:
            raise SingletonException("Tried to instantiate more than one SetupHandler object")
        SetupHandler.__instance = self

        self.__file_handler = None
        self.__audio_parameters = {}
        self.__audio_parameters = {**self.__audio_parameters, **AUDIO_PROCESSING_PARAMETERS}
        
    @staticmethod
    def get_instance() -> SetupHandler:
        """Returns the instance of the singleton, creating it if the method has never been called.

        :returns: The :py:class:`SetupHandler` instance
        """
        if SetupHandler.__instance is None:
            SetupHandler()
        return SetupHandler.__instance
    
    def setup(self) -> dict:
        """Gets the info needed as user input and fills the audio parameters dictionary accordingly,
        along with the :py:type:`AudioProducer` object. See the specific private input functions of
        this class for more info about the user input.
        
        :returns: The audio parameters dictionary filled according to user input.
        """
        user_input = self.__get_user_input()
        audio_type = user_input['audioType']
        self.__audio_parameters['audioType'] = audio_type
        self.__audio_parameters['sampleFormat'] = pyaudio.paFloat32
        self.__audio_parameters['npFormat'] = self.__get_numpy_format(pyaudio.paFloat32)
        self.__audio_parameters['instruments'] = user_input['instruments']

        if audio_type == "r":
            tracks, sr = self.__file_handler.get_tracks(user_input['songIndex'])
            self.__audio_parameters['tracks'] = tracks
            self.__audio_parameters['sampleRate'] = sr
            self.__audio_parameters['channels'] = len(tracks)
            self.__audio_parameters['audioPlayback'] = user_input['audioPlayback']
            self.__audio_producer = RecordedAudioProducer(self.__audio_parameters)
        elif audio_type == "l":
            sound_card_info = self.__get_sound_card_info(user_input['soundCardIndex'])
            self.__audio_parameters['inputDeviceIndex'] = user_input['soundCardIndex']
            self.__audio_parameters['sampleRate'] = sound_card_info['sampleRate']
            self.__audio_parameters['channels'] = sound_card_info['inChannels']
            self.__audio_parameters['audioPlayback'] = False
            self.__audio_producer = LiveAudioProducer(self.__audio_parameters)
        else:
            raise SetupException("Unrecognized audio type (was "+audio_type+")")

        return self.__audio_parameters

    def get_audio_producer(self) -> AudioProducer:
        """Getter for the :py:class:`AudioProducer` object.

        :returns: The :py:class:`AudioProducer` object.
        """
        return self.__audio_producer
    
    def set_main_path(self, main_path: str) -> None:
        """Creates the :py:class:`FileHandler` instance with a given string path.
        The path has to be fed by the main script as it is the point from here each
        relative path is calculated (see :py:class:`FileHandler` docs for more info).
        """
        self.__file_handler = FileHandler(main_path)
        self.__file_handler.unzip_files()

    def set_audio_parameters(self, audio_parameters: dict):
        """Sets the audio parameters to a given dictionary.

        :param audio_parameters: new dictionary to set.
        """
        self.__audio_parameters = audio_parameters

    def get_audio_parameters(self) -> dict:
        """Getter for the audio parameters attribute

        :returns: A :py:type:`dict` containing the audio parameters
        """
        return self.__audio_parameters
    
    def get_audio_streams(self):
        """Creates a new audio stream based on the audio parameters of the :py:class:`SetupHandler`.
        Specifically, it creates an input stream if the user chose to use live audio, or an output stream
        if the user chose to use recorded audio and wants to hear the song while it's being processed.

        :returns: A :py:type:`tuple` containing the input and output stream (:py:type:`None` if they aren't created).
        """
        params = self.__audio_parameters
        in_stream = None
        out_stream = None
        pa = pyaudio.PyAudio()

        if params['audioType'] == "l":
            in_stream = pa.open(
                format=params['sampleFormat'],
                channels=params['channels'],
                rate=params['sampleRate'],
                frames_per_buffer=params['chunkSize'],
                input_device_index=params['inputDeviceIndex'],
                input=True
            )
        
        if params['audioPlayback']:
            device = pa.get_default_output_device_info()
            out_stream = pa.open(
                rate=int(device['defaultSampleRate']),
                channels=1,
                format=pyaudio.paFloat32,
                frames_per_buffer=params['chunkSize'],
                output=True
            )

        return in_stream, out_stream

    def __get_user_input(self) -> dict:
        """Wrapper for all functions to get the user input.

        :returns: A :py:type:`dict` containing the user's choises.
        """
        audio_type = self.__get_audio_type()
        song_index = None
        audio_playback = False
        sound_card_index = None
        instruments = None
        if audio_type == "r":
            song_index = self.__get_song_index()
            audio_playback = self.__get_audio_playback()
            instruments = self.__get_instruments_for_tracks(self.__file_handler.get_number_of_tracks(song_index))
        else:
            sound_card_index = self.__get_sound_card_index()
            instruments = self.__get_instruments_for_tracks(self.__get_sound_card_info(sound_card_index)['inChannels'])
        external_osc_controller = self.__get_using_external_osc_controller()

        return {
            'audioType': audio_type,
            'songIndex': song_index,
            'soundCardIndex': sound_card_index,
            'externalOscController': external_osc_controller,
            'audioPlayback': audio_playback,
            'instruments': instruments
        }    

    def __print_console_error(self, string):
        """Prints red on the console if the user chose an unexisting option.
        :param string: String to print on the console.
        """
        print('\033[91m', string, '\033[0m')

    def __get_audio_type(self) -> str:
        """Gets the type of audio from user input. It can be either `live` or `recorded`

        :returns: a string that is either "r" or "l"
        """
        live = ["l", "live"]
        recorded = ["r", "recorded"]

        while True:
            user_input = input("\nDo you want to use live or recorded audio? (l/r): ").lower()
            if user_input in live: return "l"
            if user_input in recorded:return "r"
            self.__print_console_error("Please select a valid type (type either \"r\" or \"l\")")

    def __get_song_index(self) -> int:
        """Gets the index of the song used for recorded audio from user input.

        :returns: the index of the chosen song (depends on the number of available songs)
        """
        file_handler = self.__file_handler
        list_of_songs = file_handler.get_list_of_songs()
        number_of_songs = file_handler.get_number_of_songs()

        while True:
            print("\n[Available Songs]")
            for s in list_of_songs: 
                print(s)
            user_input = input("Please select one of the songs (type only the index): ")
            try:
                user_input = int(user_input)  
                if user_input > 0 and user_input <= number_of_songs:
                    return user_input
                self.__print_console_error("Please type only the number of the song you want to use (from 1 to "+str(number_of_songs)+")")
            except: 
                self.__print_console_error("Please enter a valid number")

    def __get_audio_playback(self) -> bool:
        """Asks the user if he wants to hear the song that is being processed when using recorded audio.

        :returns: A :py:type:`bool` set to `True` if the user wants to hear the song
        """
        yes = ['y', 'yes']
        no = ['n', 'no']

        while True:
            user_input = input("\nDo you want to hear the song as its being processed? (y/n): ")
            if user_input in yes: return True
            if user_input in no: return False
            self.__print_console_error("Please input a valid string (\"y\", \"yes\" or \"n\", \"no\")")

    def __get_sound_card_index(self) -> int:
        """Gets the sound card index for live audio from user input.

        :returns: the index of the chosen sound card (depends on the number of available sound cards)
        """
        default = ["d", "default"]

        pa = pyaudio.PyAudio()
        info = pa.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        usable_devices = []
        sound_card_list = []

        for i in range(0, numdevices):
            if (pa.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                sound_card_list.append("("+str(i)+") "+pa.get_device_info_by_host_api_device_index(0, i).get('name'))
                usable_devices.append(i)

        while True:
            print("\n[Available Input Soundcards]")
            for s in sound_card_list:
                print(s)
            user_input = input("Please select your preferred input sound card (type only the index or \"default\"): ")
            try:
                user_input = int(user_input)
                if user_input in usable_devices: return user_input
                self.__print_console_error("Please type only the number of the song you want to use (from 0 to "+str(numdevices)+")")
            except:
                self.__print_console_error("Please enter a valid number")

    def __get_sound_card_info(self, sound_card_index) -> dict:
        """Gets the info of a sound card given its index in the Host API using :py:class:`PyAudio`.

        :returns: a dictionary containing the sound card's info.
        """
        pa = pyaudio.PyAudio()
        sound_card_info = pa.get_device_info_by_host_api_device_index(0, sound_card_index)
        return {
            'sampleRate': int(sound_card_info['defaultSampleRate']),
            'inChannels': int(sound_card_info['maxInputChannels']),
            'outChannels': int(sound_card_info['maxOutputChannels'])
        }
    
    def __get_instruments_for_tracks(self, channels) -> list:
        """Gets the instruments for each of the tracks.

        :returns: a :py:type:`list` containing an instrument type for each track.
        """
        confirm = ["", "y", "yes", "c", "confirm"]
        instruments = []
        for _ in range(channels): 
            instruments.append(Instruments.DEFAULT)

        while True:
            print("\n===== INSTRUMENTS SELECTION =====")
            for i in range(len(instruments)):
                print("("+str(i)+") "+instruments[i].get_string())
            track_number = input("Choose the track for which you want to assign an instrument (from 0 to "+str(channels-1)+", leave blank to save): ")

            if track_number in confirm: return instruments

            try:
                track_number = int(track_number)
            except:
                self.__print_console_error("Please select a valid track")
                continue

            if track_number < 0 or track_number > channels-1:
                self.__print_console_error("Please select a valid track")
                continue

            print("\n== Available Instruments ==")
            for i in Instruments:
                print("("+str(i.value)+") "+i.name)
            inst_index = input("Choose an instrument for track "+str(track_number)+" (type only the corresponding index): ")

            try:
                instrument = Instruments.from_index(int(inst_index))
            except:
                self.__print_console_error("Please select a valid instrument")
                continue
            
            instruments[track_number] = instrument

    def __get_using_external_osc_controller(self) -> bool:
        """Asks the user if he wants to use an external osc controller.

        :returns: A :py:type:`bool` set to `True` if the user wants to use an external osc controller
        """

        yes = ["y", "yes"]
        no = ["n", "no"]

        while True:
            user_input = input("\nUsing external osc controller? (y/n): ")
            if user_input in yes: return True
            if user_input in no: return False
            self.__print_console_error("Please input a valid string (\"y\", \"yes\" or \"n\", \"no\")")

    def __get_numpy_format(self, pyaudio_sample_format: int):
        """Returns the corresponding `numpy` format to a given `pyaudio` format.

        :param pyaudio_sample_format: pyaudio format. 
        """
        if pyaudio_sample_format == pyaudio.paInt16 : return np.int16
        if pyaudio_sample_format == pyaudio.paInt32 : return np.int32
        if pyaudio_sample_format == pyaudio.paFloat32 : return np.float32
        raise SetupException("Incompatible Sample Format")

        

class FileHandler:
    """Handles files, specifically those related to the recorded songs wavs and zips.
    """

    def __init__(self, main_path: str):
        """Creates a new :py:class:`FileHandler` and sets the path for the zips folder and
        unzipped songs folder.

        :param main_path: Path of the main script from which the relative paths of resources are calculated
        """
        self._source_folder_path = os.path.join(main_path, "resources/test_songs")
        self._dest_folder_path = os.path.join(main_path, "resources/test_songs_unzipped")
        self._zip_extension = ".zip"
        self._number_of_songs = 0
        self._song_index_to_path_dict = {}

    def unzip_files(self):
        """Unzips any zip file found in the `resources/test_songs` folder. If the zip file has already been
        unzipped, it skips it
        """
        for item in os.listdir(self._source_folder_path):
            if item.endswith(self._zip_extension):
                self._number_of_songs += 1
                path_to_zip_file = self._source_folder_path + "/" + item
                path_to_unzipped_file = self._dest_folder_path + "/" + item
                path_to_unzipped_file = path_to_unzipped_file[:len(path_to_unzipped_file)-4]
                self._song_index_to_path_dict[self._number_of_songs] = path_to_unzipped_file
                if os.path.isdir(path_to_unzipped_file): continue
                with zipfile.ZipFile(path_to_zip_file, "r") as zip:
                    zip.extractall(self._dest_folder_path)
        if self._number_of_songs == 0:
            raise FileHandlingException("No files to unzip")

    def get_tracks(self, song_index: int) -> tuple:
        """Gets all the tracks of a song as numpy arrays, ready to be processed.
        Each track has its own stereo .wav file, so each file has to be read and converted
        to mono.

        :returns: A :py:type:`list` of :py:type:`np.ndarray` containing the tracks and the sample
        rate of the wave file read (assuming all files of the song have the same sample rate)
        """
        path = self._song_index_to_path_dict.get(song_index)
        tracks = []
        track_counter = 0
        sr = 44100

        for filename in glob.glob(os.path.join(path, '*.wav')):
            sr, track_stereo_array = read(filename)
            track_stereo_array = track_stereo_array.astype(dtype=np.float32, order='C') / 32768.0
            track_array = (track_stereo_array[:, 0] + track_stereo_array[:, 1]) / 2.0
            tracks.append(track_array)
            track_counter += 1

        return tracks, sr
    
    def get_number_of_tracks(self, song_index: int) -> int:
        """Gets the number of tracks given the index of a song

        :returns: The number of tracks of a song
        """
        path = self._song_index_to_path_dict.get(song_index)
        track_counter = 0

        for _ in glob.glob(os.path.join(path, '*.wav')):
            track_counter += 1

        return track_counter
    def get_list_of_songs(self):
        """Gets the list of available songs so that the user can chose the one he prefers.

        :returns: A :py:type:`list` of strings containing the name if the songs and their indexes.

        :raises FileHandlingException: if no songs have been found
        """
        list_of_test_songs = []
        for key in self._song_index_to_path_dict.keys():
            song_name = self._song_index_to_path_dict[key].replace(self._dest_folder_path+"/", '')
            list_of_test_songs.append("("+str(key) + ") " + song_name)
        if len(list_of_test_songs) == 0:
            raise FileHandlingException("No songs found")
        return list_of_test_songs

    def get_number_of_songs(self) -> int:
        """Getter for the number of songs attribute
        """
        return self._number_of_songs

    def get_source_folder_path(self) -> str:
        """Getter for the source folder path attribute
        """
        return self._source_folder_path
    
    def get_dest_folder_path(self) -> str:
        """Getter for the destination folder path attribute
        """
        return self._dest_folder_path