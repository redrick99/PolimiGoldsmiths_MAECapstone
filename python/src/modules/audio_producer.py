from abc import ABC, abstractmethod
import pyaudio
import numpy as np
import time
from modules.custom_exceptions import *


class AudioProducer(ABC):

    def __init__(self, parameters: dict):
        self._sample_rate = parameters['sampleRate']
        self._chunk_size = parameters['chunkSize']
        self._sample_format = parameters['sampleFormat']
        self._np_format = parameters['npFormat']
        self._channels = parameters['channels']

    @abstractmethod
    def get_next_chunk(self, in_stream: pyaudio.Stream, out_stream: pyaudio.Stream) -> list:
        pass


class LiveAudioProducer(AudioProducer):

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        
    def get_next_chunk(self, in_stream: pyaudio.Stream, out_stream: pyaudio.Stream) -> list:
        data_per_channel = []

        if in_stream is None: 
            raise AudioProducingException("Input stream was None")
        
        chunk_bytes = in_stream.read(self._chunk_size, False)
        chunk_array = np.frombuffer(chunk_bytes, dtype=self._np_format)

        for i in range(self._channels):
            data_per_channel.append(chunk_array[i::self._channels])

        return data_per_channel


class RecordedAudioProducer(AudioProducer):

    def __init__(self, parameters: dict):
        super().__init__(parameters)

        self._audio_input_tracks = parameters['tracks']
        self._audio_playback = parameters['audioPlayback']

    def get_next_chunk(self, in_stream: pyaudio.Stream, out_stream: pyaudio.Stream) -> list:
        data_per_channel = []
        cs = self._chunk_size
        
        for i in range(len(self._audio_input_tracks)):
            if len(self._audio_input_tracks[i]) == 0: raise FinishedSongException("Tried to access finished song")
            if len(self._audio_input_tracks[i]) < cs:
                data_per_channel.append(self._audio_input_tracks[i].copy())
                self._audio_input_tracks[i] = []
                continue
            data_per_channel.append(self._audio_input_tracks[i][0:cs])
            self._audio_input_tracks[i] = self._audio_input_tracks[i][cs:len(self._audio_input_tracks[i])]

        if out_stream:
            self.__play_audio_chunk(data_per_channel, out_stream)
        else:
            time.sleep(np.abs(cs/self._sample_rate))

        return data_per_channel

    def __play_audio_chunk(self, data_per_channel: list, out_stream: pyaudio.Stream):
        np_data = np.array(data_per_channel, dtype=np.float32, copy=True)
        np_data = np_data.sum(axis=0) / float(len(data_per_channel))

        out_stream.write(np_data.tobytes(order='C'))