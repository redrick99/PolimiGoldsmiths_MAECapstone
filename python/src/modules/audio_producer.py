from abc import ABC, abstractmethod
import pyaudio
import numpy as np
import time
from modules.custom_exceptions import *


class AudioProducer(ABC):
    """Abstract class representing an input audio handler. Can be inherited to create new ways of producing audio.
    """

    def __init__(self, parameters: dict):
        """Constructor for the AudioProducer class.

        **Args:**

        `parameters`: A dictionary containing audio parameters.

        **Class Attributes:**

        `_sample_rate`: Sample rate at which to read and write audio.

        `_chunk_size`: Size of the chunk of audio to read.

        `_sample_format`: Format at which to read and write audio (float, int, ...).

        `_np_format`: Format of numpy used for conversions.

        `_channels`: Number of channels of the audio source.
        """
        self._sample_rate = parameters['sampleRate']
        self._chunk_size = parameters['chunkSize']
        self._sample_format = parameters['sampleFormat']
        self._np_format = parameters['npFormat']
        self._channels = parameters['channels']

    @abstractmethod
    def get_next_chunk(self, in_stream: pyaudio.Stream, out_stream: pyaudio.Stream) -> list:
        """Abstract function to get the next chunk of audio from an input stream.

        **Args:**

        `in_stream`: Input stream to use for live audio applications.

        `out_stream`: Output stream used to playback audio in recorded applications.

        **Returns:**

        A representation of the next chunk of audio read by the input device.
        """
        pass


class LiveAudioProducer(AudioProducer):
    """Subclass of AudioProducer that handles input audio taken live from an input sound card.
    """

    def __init__(self, parameters: dict):
        """Constructor for the LiveAudioProducer class.

        **Args:**

        `parameters`: A dictionary containing audio parameters.
        """
        super().__init__(parameters)
        
    def get_next_chunk(self, in_stream: pyaudio.Stream, out_stream: pyaudio.Stream) -> list:
        """Gets the next chunk of audio from the input stream.

        **Args:**

        `in_stream`: Input stream to use for live audio applications.

        `out_stream`: Output stream used to playback audio in recorded applications.

        **Returns:**

        The next chunk of audio read by the input stream, divided for each track of the input sound card.
        """
        data_per_channel = []

        if in_stream is None: 
            raise AudioProducingException("Input stream was None")
        
        chunk_bytes = in_stream.read(self._chunk_size, False)  # Reads from input stream
        chunk_array = np.frombuffer(chunk_bytes, dtype=self._np_format)  # Converts to numpy array

        for i in range(self._channels):  # Divides master audio chunk into channels
            data_per_channel.append(chunk_array[i::self._channels])

        return data_per_channel


class RecordedAudioProducer(AudioProducer):
    """Subclass of AudioProducer that handles recorded audio taken from wav audio files.
    """

    def __init__(self, parameters: dict):
        """Constructor for the RecordedAudioProducer class

        **Args:**

        `parameters`: A dictionary containing audio parameters

        **Class Attributes:**

        `_audio_input_tracks`: Number of tracks of the audio, corresponding to the number of audio files contained in
        a song's folder.

        `_audio_playback`: Whether the user wants to play back the song as it's being processed.
        """
        super().__init__(parameters)

        self._audio_input_tracks = parameters['tracks']
        self._audio_playback = parameters['audioPlayback']

    def get_next_chunk(self, in_stream: pyaudio.Stream, out_stream: pyaudio.Stream) -> list:
        """Gets the next chunk of audio from audio files of tracks.

        **Args:**

        `in_stream`: Unused (not needed with recorded audio).

        `out_stream`: Output stream used to playback audio in recorded applications.

        **Returns:**

        The next chunk of audio read by the input stream, divided for each track (which are stored in
        different files).
        """
        data_per_channel = []
        cs = self._chunk_size
        
        for i in range(len(self._audio_input_tracks)):
            if len(self._audio_input_tracks[i]) == 0:
                raise FinishedSongException("Tried to access finished song")
            if len(self._audio_input_tracks[i]) < cs:
                data_per_channel.append(self._audio_input_tracks[i].copy())
                self._audio_input_tracks[i] = []
                continue
            data_per_channel.append(self._audio_input_tracks[i][0:cs])
            self._audio_input_tracks[i] = self._audio_input_tracks[i][cs:len(self._audio_input_tracks[i])]

        if out_stream:  # Plays audio if the user choose to do so during startup
            self.__play_audio_chunk(data_per_channel, out_stream)
        else:  # Waits for chunk_size/sample_rate otherwise (to mimic live audio behaviour)
            time.sleep(np.abs(cs/self._sample_rate))

        return data_per_channel

    def __play_audio_chunk(self, data_per_channel: list, out_stream: pyaudio.Stream):
        """Plays a given chunk of audio on an output stream.

        **Args:**

        `data_per_channel`: Audio chunk to be played, split into tracks.

        `out_stream`: Output stream used to play the audio chunk.
        """
        np_data = np.array(data_per_channel, dtype=np.float32, copy=True)
        np_data = np_data.sum(axis=0) / float(len(data_per_channel))

        out_stream.write(np_data.tobytes(order='C'))
