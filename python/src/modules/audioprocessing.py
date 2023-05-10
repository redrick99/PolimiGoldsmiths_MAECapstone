from abc import ABC, abstractmethod
import multiprocessing as mp
import warnings
import numpy as np
import librosa
import modules.default_parameters as dp
from scipy.signal import find_peaks
from modules.utilities import *
from modules.connection import OSCConnectionHandler, LFAudioMessage, HFAudioMessage


class AudioProcessor(ABC):
    """Abstract class to handle all Audio Processing methods and functions
    """

    def __init__(self, parameters: dict):
        """Constructor with all audio parameters needed fot processing.
        """
        self._sample_rate = parameters['sampleRate']
        self._chunk_size = parameters['chunkSize']
        self._np_format = parameters['npFormat']
        self._nfft = parameters['nfft']
        self._hop_length = parameters['hopLength']
        self._window_size = parameters['winSize']
        self._window_type = parameters['winType']
        self._p_threshold = parameters['pitchThreshold']
        self._normType = parameters['normType']
        
    @abstractmethod
    def process(self, frame, inst: Instruments):
        """Abstract method to process the given audio frame.

        :param frame: Chunk of audio to process
        :param inst: Instrument of the track from which to get the frequency range
        """
        pass

    def _compute_stft(self, frame):
        """Computes the STFT of a given signal frame, with the object's parameters.

        :param frame: Chunk of audio to process
        """
        return np.abs(librosa.stft(y=frame, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type, dtype=self._np_format))

    def _get_mono_frequency(self, frame):
        """Returns the peak frequency of the audio frame (Monophonic Pitch Detection).

        :param frame: Chunk of audio to process
        """
        spectrum = np.abs(np.fft.rfft(frame))
        peaks, _ = find_peaks(spectrum, distance=10)
        
        if len(peaks) == 0:
            return 0
        
        peak_index = np.argmax(spectrum[peaks])
        return peaks[peak_index] * self._sample_rate / self._sample_rate

    def _get_poly_frequencies(self, signal_stft, inst: Instruments):
        """Returns the peak frequencies of the audio frame from its spectrum (Polyphonic Pitch Detection)

        :param signal_stft: Spectrum of a frame of the signal
        """
        freq_range = inst.get_fundamental_frequency_range()
        if range is None:
            return [0]

        pitches, magnitudes = librosa.piptrack(S=signal_stft, sr=self._sample_rate, fmin=freq_range[0], fmax=freq_range[1], threshold=self._p_threshold)
        indexes = np.argsort(magnitudes[:, 0])[::-1]
        pitches = pitches[indexes, 0]

        return pitches[pitches > 0]

    def _get_spectral_centroid(self, signal_stft):
        """
        Returns the spectral centroid of the audio frame from its spectrum

        :param signal_stft: Spectrum of a frame of the signal
        """
        cent = librosa.feature.spectral_centroid(S=signal_stft, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        cent = np.mean(cent, axis=1)
        return cent

    def _get_spectral_bandwidth(self, signal_stft):
        """Returns the spectral bandwidth of the audio frame from its spectrum

        :param signal_stft: Spectrum of a frame of the signal
        """
        sb = librosa.feature.spectral_bandwidth(S=signal_stft, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        sb = np.mean(sb, axis=1)
        return sb

    def _get_spectral_contrast(self, signal_stft):
        """
        Returns the spectral contrast of the audio frame from its spectrum

        :param signal_stft: Spectrum of a frame of the signal
        """
        sc = librosa.feature.spectral_contrast(S=signal_stft, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        return np.mean(sc, axis=1)

    def _get_spectral_flatness(self, signal_stft):
        """Returns the spectral flatness of the audio frame from its spectrum

        :param signal_stft: Spectrum of a frame of the signal
        """
        sf = librosa.feature.spectral_flatness(S=signal_stft, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        return np.mean(sf, axis=1)

    def _get_spectral_rolloff(self, signal_stft):
        """Returns the spectral rolloff of the audio frame from its spectrum

        :param signal_stft: Spectrum of a frame of the signal
        """
        sr = librosa.feature.spectral_rolloff(S=signal_stft, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        return np.mean(sr, axis=1)
    
    def _normalize(self, array) -> np.ndarray:
        """Implements different normalizations based on the current selected normalization type

        :param array: Array to normalize

        :returns: The normalized array (or the array as is if normalization is set to NONE)
        """
        if self._normType == Normalizations.NONE:
            return array
        
        if self._normType == Normalizations.PEAK:
            return array / np.max(np.abs(array))
        
        if self._normType == Normalizations.RMS:
            return np.sqrt(np.mean(np.square(array)))
        
        if self._normType == Normalizations.Z_SCORE:
            return (array - np.mean(array)) / np.std(array)
        
        if self._normType == Normalizations.MIN_MAX:
            min_value = np.amin(array)
            max_value = np.amax(array)
            return (array - min_value) / (max_value - min_value)


class DefaultAudioProcessor(AudioProcessor):
    """Implements the default chain used to process low level features
    """

    def __init__(self, parameters: dict):
        """Constructor
        """
        super().__init__(parameters)

    def process(self, frame, inst: Instruments):
        """Processes the given audio frame according to a defined chain

        :param frame: Chunk of audio to process
        :param inst: Instrument of the track from which to get the frequency range

        :returns: Low Level Features as floats ordered inside an array
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = self._normalize(array=frame)
            signal_stft = self._compute_stft(frame)
            pitches = np.round(self._get_poly_frequencies(signal_stft, inst))
            spec_centroid = np.round(self._get_spectral_centroid(signal_stft)[0])
            spec_bandwidth = np.round(self._get_spectral_bandwidth(signal_stft)[0])
            spec_flatness = self._get_spectral_flatness(signal_stft)[0]
            spec_rolloff = np.round(self._get_spectral_rolloff(signal_stft)[0])

        return [spec_centroid, spec_bandwidth, spec_flatness, spec_rolloff, *pitches[:4]]


class InputHandler(ABC):
    """The InputHandler abstract class declares a set of methods for processing data
    """

    def __init__(self, parameters: dict, channel: int, instrument: Instruments):
        """Superclass Constructor

        :param parameters: Audio parameters used to process the audio
        :param channel: Channel index of the track to process assigned to this instance
        :param instrument: Instrument of the track to process assigned to this instance
        """
        net_params = dp.NET_PARAMETERS
        self._connection_handler = OSCConnectionHandler.get_instance(net_params['outNetAddress'], net_params['outNetPort'])
        self._signal_threshold = parameters['signalThreshold']
        self.__instrument = instrument
        self._lock = mp.Lock()
        self.channel = channel
        self.__priority = 0

    @abstractmethod
    def process(self, data):
        """Abstract method to implement the processing chain

        :param data: Data to process
        """
        pass

    def set_instrument(self, instrument: Instruments):
        """Synchronized setter for the "instrument" attribute
        """
        self._lock.acquire()
        self.__instrument = instrument
        self._lock.release()

    def get_instrument(self) -> Instruments:
        """Synchronized getter for the "instrument" attribute
        """
        self._lock.acquire()
        inst = self.__instrument
        self._lock.release()
        return inst
    
    def set_priority(self, priority: int):
        """Synchronized setter for the "priority" attribute
        """
        self._lock.acquire()
        self.__priority = priority
        self._lock.release()

    def get_priority(self) -> int:
        """Synchronized getter for the "priority" attribute
        """
        self._lock.acquire()
        p = self.__priority
        self._lock.release()
        return p
    
    def handle_settings(self, settings):
        """Handles incoming settings

        :param settings: settings to handle
        """
        pass


class LFAudioInputHandler(InputHandler):
    """Handles the Low Level feature processing of a channel of either live or recorded audio.
    """

    def __init__(self, parameters: dict, channel: int, instrument: Instruments):
        """Constructor

        :param parameters: Audio parameters used to process the audio
        :param channel: Channel index of the track to process assigned to this instance
        :param instrument: Instrument of the track to process assigned to this instance
        """
        super().__init__(parameters, channel, instrument)
        self._audio_processor = DefaultAudioProcessor(parameters)

    def __no_signal(self, data):
        """Checks if there's an actual signal inside the audio frame by computing the rms and evaluating it against a predefined threshold.

        :param data: Data frame to check

        :returns: A :py:type:`bool` set to `True` if the signal is virtually non-existent.
        """
        rms = np.sqrt(np.mean(np.square(data)))
        return rms <= self._signal_threshold
    
    def process(self, data):
        """Processes an audio frame or returns if the frame contains no information

        :param data: audio frame to process
        """
        if self.__no_signal(data):
            return

        inst = self.get_instrument()
        processed_data = self._audio_processor.process(data, inst)
        print_data(self.channel, processed_data)
        msg = LFAudioMessage(processed_data, self.channel, inst)
        self._connection_handler.send_message(msg)


class HFAudioInputHandler(InputHandler):
    """Handles the High Level Features Processing
    """

    def __init__(self, parameters: dict, channel: int, instrument: Instruments):
        """Constructor

        :param parameters: Audio parameters used to process the audio
        :param channel: Channel index of the track to process assigned to this instance
        :param instrument: Instrument of the track to process assigned to this instance
        """
        super().__init__(parameters, channel, instrument)
        self.__number_of_samples = parameters['hfNumberOfSamples']
        self.__np_format = parameters['npFormat']
        self.__data = np.array([], dtype=self.__np_format)
    
    def process(self, data):
        """Processes an audio frame for High Level features

        :param data: audio frame to process
        """
        if len(self.__data) < self.__number_of_samples:
            data_array = np.array(data, dtype=self.__np_format, copy=True)
            data_array = data_array.sum(axis=0) / float(len(data))
            self.__data = np.concatenate((self.__data, data_array), axis=0)
            return
        
        data_to_process = self.__data[0:self.__number_of_samples].copy()
        self.__data = np.array([], dtype=self.__np_format)
        
        # TODO actually process the audio

        msg = HFAudioMessage("test", self.channel)
        self._connection_handler.send_message(msg)
