from abc import ABC, abstractmethod
import os
import logging
import multiprocessing as mp
import warnings
import numpy as np
import librosa
import tensorflow as tf
import modules.default_parameters as dp
from scipy.signal import find_peaks
from modules.utilities import *
from modules.connection import OSCConnectionHandler, LFAudioMessage, HFAudioMessage

# Removes Tensorflow logs and warnings
tf.keras.utils.disable_interactive_logging()
tf.get_logger().setLevel(logging.ERROR)


class AudioProcessor(ABC):
    """Abstract class to handle all Audio Processing methods and functions. Can be inherited to implement custom
    processing methods.
    """

    def __init__(self, parameters: dict):
        """Constructor for the AudioProcessor class.

        **Args:**

        `parameters`: A dictionary containing audio parameters.

        **Class Attributes:**

        '_sample_rate`: Sample rate at which to read and write audio.

        `_chunk_size`: Size of the chunk of audio to read.

        '_np_format`: Numpy format used to process audio.

        `_nfft`: Size of the FFT used during processing.

        '_hop_length`: Hop Length of the FFT used during processing.

        `_window_size`: Window Size of the FFT used during processing.

        '_window_type`: Window Type of the FFT used during processing.

        `_p_threshold`: Threshold under which the frequency component of the audio piece processed during polyphonic
        pitch extraction is discarded.

        '_normType`: Normalization type used during processing.
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
        """Abstract method to process a given audio frame.

        **Args:**

        `frame`: Chunk of audio to process.

        `inst`: Instrument assigned to the track (used to get the frequency range for audio processing).
        """
        pass

    def _compute_stft(self, frame):
        """Computes the STFT of a given signal frame, with the object's parameters.

        **Args:**

        `frame`: Audio frame to process.
        """
        return np.abs(librosa.stft(y=frame, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type, dtype=self._np_format))

    def _get_mono_frequency(self, frame):
        """Returns the peak frequency of the audio frame (Monophonic Pitch Detection).

        **Args:**

        `frame`: Audio frame to process.
        """
        spectrum = np.abs(np.fft.rfft(frame))
        peaks, _ = find_peaks(spectrum, distance=10)
        
        if len(peaks) == 0:
            return 0
        
        peak_index = np.argmax(spectrum[peaks])
        return peaks[peak_index] * self._sample_rate / self._sample_rate

    def _get_poly_frequencies(self, signal_stft, inst: Instruments):
        """Returns the peak frequencies of the audio frame from its spectrum (Polyphonic Pitch Detection).

        **Args:**

        `signal_stft`: Spectrum of the frame to process.

        `inst`: Instrument that generated the processed audio frame.
        """
        freq_range = inst.get_fundamental_frequency_range()
        if range is None:
            return [0]

        pitches, magnitudes = librosa.piptrack(S=signal_stft, sr=self._sample_rate, fmin=freq_range[0], fmax=freq_range[1], threshold=self._p_threshold)
        indexes = np.argsort(magnitudes[:, 0])[::-1]
        pitches = pitches[indexes, 0]

        return pitches[pitches > 0]

    def _get_spectral_centroid(self, signal_stft):
        """Returns the mean value of the spectral centroid of the audio frame from its spectrum.

        **Args:**

        `signal_stft`: Spectrum of the frame to process.
        """
        cent = librosa.feature.spectral_centroid(S=signal_stft, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        cent = np.mean(cent, axis=1)
        return cent

    def _get_spectral_bandwidth(self, signal_stft):
        """Returns the mean value of the spectral bandwidth of the audio frame from its spectrum.

        **Args:**

        `signal_stft`: Spectrum of the frame to process.
        """
        sb = librosa.feature.spectral_bandwidth(S=signal_stft, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        sb = np.mean(sb, axis=1)
        return sb

    def _get_spectral_contrast(self, signal_stft):
        """
        Returns the mean value of the spectral contrast of the audio frame from its spectrum.

        **Args:**

        `signal_stft`: Spectrum of the frame to process.
        """
        sc = librosa.feature.spectral_contrast(S=signal_stft, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        return np.mean(sc, axis=1)

    def _get_spectral_flatness(self, signal_stft):
        """Returns the mean value of the spectral flatness of the audio frame from its spectrum.

        **Args:**

        `signal_stft`: Spectrum of the frame to process.
        """
        sf = librosa.feature.spectral_flatness(S=signal_stft, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        return np.mean(sf, axis=1)

    def _get_spectral_rolloff(self, signal_stft):
        """Returns the spectral rolloff of the audio frame from its spectrum.

        **Args:**

        `signal_stft`: Spectrum of the frame to process.
        """
        sr = librosa.feature.spectral_rolloff(S=signal_stft, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        return np.mean(sr, axis=1)
    
    def _normalize(self, array) -> np.ndarray:
        """Implements different normalizations based on the current selected normalization type.

        **Args:**

        `array`: Array to normalize

        **Returns:**

        The normalized array (or the array as is if normalization is set to NONE).
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
    """Implements the default chain used to process low level features.
    """

    def __init__(self, parameters: dict):
        """Constructor for the DefaultAudioProcessor class.

        **Args:**

        `parameters`: A dictionary containing audio parameters.
        """
        super().__init__(parameters)

    def process(self, frame, inst: Instruments):
        """Processes the given audio frame according to a defined chain.

        **Args:**

        `frame`: Audio frame to process.

        `inst`: Instrument of the track from which to get the frequency range.

        **Returns:**

        Low Level Features as floats ordered inside an array.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # frame = self._normalize(array=frame)
            frame = librosa.util.normalize(frame)
            signal_stft = self._compute_stft(frame)
            pitches = np.round(self._get_poly_frequencies(signal_stft, inst))
            spec_centroid = np.round(self._get_spectral_centroid(signal_stft)[0])
            spec_bandwidth = np.round(self._get_spectral_bandwidth(signal_stft)[0])
            spec_flatness = self._get_spectral_flatness(signal_stft)[0]
            spec_rolloff = np.round(self._get_spectral_rolloff(signal_stft)[0])

        return [spec_centroid, spec_bandwidth, spec_flatness, spec_rolloff, *pitches[:4]]


class InputHandler(ABC):
    """The InputHandler abstract class declares a set of methods for processing data and extract features.
    Can be inherited to implement custom methods to process data and extract features.
    """

    def __init__(self, parameters: dict, channel: int, instrument: Instruments):
        """Constructor for the InputHandler class.

        **Args:**

        `parameters`: Audio parameters used to process the audio.

        `channel`: Channel index of the track to process assigned to this instance.

        `instrument`: Instrument of the track to process assigned to this instance.

        **Class Attributes:**

        `_connection_handler`: Object used to send features to external applications.

        `_signal_threshold`: Threshold under which the signal is considered to be null.

        `__instrument`: Instrument of the source of audio.

        `_lock`: Mutex lock used for synchronization purposes.

        `channel`: Number assigned to the track that this object is currently processing.

        `__priority`: Number to indicate the priority value of an instance with respect to the other instances.
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
        """Abstract method to implement the processing and feature extraction chain.

        **Args:**
        `data`: Data to process.
        """
        pass

    def set_instrument(self, instrument: Instruments):
        """Synchronized setter for the `__instrument` attribute.

        **Args:**

        `instrument`: New instrument to assign.
        """
        self._lock.acquire()
        self.__instrument = instrument
        self._lock.release()

    def get_instrument(self) -> Instruments:
        """Synchronized getter for the `__instrument` attribute.
        """
        self._lock.acquire()
        inst = self.__instrument
        self._lock.release()
        return inst
    
    def set_priority(self, priority: int):
        """Synchronized setter for the `__priority` attribute.

        **Args:**

        `priority`: New priority to assign.
        """
        self._lock.acquire()
        self.__priority = priority
        self._lock.release()

    def get_priority(self) -> int:
        """Synchronized getter for the `__priority` attribute.
        """
        self._lock.acquire()
        p = self.__priority
        self._lock.release()
        return p

    def _no_signal(self, data):
        """Checks if there's an actual signal inside the audio frame by computing the rms and evaluating it against a
        predefined threshold.

        **Args:**

        `data`: Data frame to check

        **Returns:**

        A `bool` set to `True` if the signal is virtually non-existent.
        """
        rms = np.sqrt(np.mean(np.square(data)))
        return rms <= self._signal_threshold
    
    def handle_settings(self, settings):
        """Handles incoming settings.

        **Args:**

        `settings`: settings to handle.
        """
        pass


class LFAudioInputHandler(InputHandler):
    """Handles the Low Level feature processing of a channel of either live or recorded audio.
    """

    def __init__(self, parameters: dict, channel: int, instrument: Instruments):
        """Constructor for the LFAudioInputHandler class.

        **Args:**

        `parameters`: Audio parameters used to process the audio.

        `channel`: Channel index of the track to process assigned to this instance.

        `instrument`: Instrument of the track to process assigned to this instance.

        **Class Attributes:**

        `_audio_processor`: Processor object used to process audio.
        """
        super().__init__(parameters, channel, instrument)
        self._audio_processor = DefaultAudioProcessor(parameters)
    
    def process(self, data):
        """Processes an audio frame or returns if the frame has virtually no signal.

        :param data: audio frame to process
        """
        if self._no_signal(data):
            return

        inst = self.get_instrument()
        processed_data = self._audio_processor.process(data, inst)
        # print_data(self.channel, processed_data)
        msg = LFAudioMessage(processed_data, self.channel, inst)
        self._connection_handler.send_message(msg)


class HFAudioInputHandler(InputHandler):
    """Handles the High Level feature processing of either live or recorded audio.
    """

    def __init__(self, parameters: dict, channel: int, instrument: Instruments):
        """Constructor of the HFAudioInputHandler class.

        **Args:**

        `parameters`: Audio parameters used to process the audio.

        `channel`: Unused (high-level features are computed over the sum of all tracks).

        `instrument`: Unused (high-level features are computed over the sum of all tracks).

        **Class Attributes:**

        `__arousal_values`: Array containing previous arousal values used to compute a moving average.

        `__valence_values`: Array containing previous valence values used to compute a moving average.

        '__nn_model': Neural network model used to extract the mood from the piece of audio.
        """
        super().__init__(parameters, channel, instrument)
        path = os.path.join(parameters['mainPath'], "resources", "nn_models", "modelv5.h5")
        average_length = parameters['hfMovingAverageLengthInSeconds']
        self.__arousal_values = np.zeros(int(average_length/0.5))
        self.__valence_values = np.zeros(int(average_length/0.5))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.__nn_model = tf.keras.models.load_model(path)
    
    def process(self, data):
        """Processes an audio frame for High Level features.

        **Args:**

        `data`: Piece of audio to process.
        """
        if self._no_signal(data):
            return

        data = librosa.util.normalize(data)
        data_tensor = np.copy(data)
        data_tensor = np.expand_dims(data_tensor, axis=0)

        prediction = self.__nn_model.predict(data_tensor)[0]

        self.__arousal_values[0] = prediction[0]
        self.__valence_values[0] = prediction[1]
        self.__arousal_values = np.roll(self.__arousal_values, -1)
        self.__valence_values = np.roll(self.__valence_values, -1)
        prediction[0] = np.average(self.__arousal_values)
        prediction[1] = np.average(self.__valence_values)

        print_data_alt_color(channel=1, data=prediction)

        msg = HFAudioMessage(prediction, 0)
        self._connection_handler.send_message(msg)
