from abc import ABC, abstractmethod
import time, multiprocessing as mp, numpy as np, librosa, scipy.signal as scis
import modules.utilities as u
from modules.connection import OSCConnectionHandler, LFAudioMessage, HFAudioMessage

class AudioProcessor(ABC):
    """
    Abstract class to handle all Audio Processing methods and functions
    """

    def __init__(self, sample_rate, chunk_size, np_format, nfft, hop_length, window_size, window_type, pitch_threshold):
        """
        Constructor with all audio parameters needed fot processing
        """
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._np_format = np_format
        self._nfft = nfft
        self._hop_length = hop_length
        self._window_size = window_size
        self._window_type = window_type
        self._p_threshold = pitch_threshold
        
        self._debugger = u.Debugger()

    @abstractmethod
    def process(self, frame, inst: u.Instruments):
        """
        Abstract method to process the given audio frame
        """
        pass

    def _compute_stft(self, frame):
        """
        Computes the STFT of a given signal, with the object's parameters
        """
        return np.abs(librosa.stft(y=frame, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type, dtype=self._np_format))

    def _get_mono_frequency(self, frame):
        """
        Returns the peak frequency of the audio frame (Monophonic Pitch Detection)

        Parameters:
        - frame: chunk of audio to process
        """
        spectrum = np.abs(np.fft.rfft(frame))
        peaks, _ = scis.find_peaks(spectrum, distance=10)
        
        if len(peaks) == 0:
            return 0
        
        peak_index = np.argmax(spectrum[peaks])
        return peaks[peak_index] * self._sample_rate / self._sample_rate

    def _get_poly_frequencies(self, S, inst: u.Instruments):
        """
        Returns the peak frequencies of the audio frame from its spectrum (Polyphonic Pitch Detection)

        Parameters:
        - S: spectrum of a signal
        """
        range = u.get_fundamental_frequency_range_by_instrument(inst)
        if range is None: return [0]

        pitches, magnitudes = librosa.piptrack(S=S, sr=self._sample_rate, fmin=range[0], fmax=range[1], threshold=self._p_threshold)
        indexes = np.argsort(magnitudes[:, 0])[::-1]
        pitches = pitches[indexes, 0]

        return pitches[pitches > 0]

    def _get_spectral_centroid(self, S):
        """
        Returns the spectral centroid of the audio frame from its spectrum

        Parameters:
        - S: spectrum of a signal
        """
        cent = librosa.feature.spectral_centroid(S=S, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        cent = np.mean(cent, axis=1)
        return cent

    def _get_spectral_bandwidth(self, S):
        """
        Returns the spectral bandwidth of the audio frame from its spectrum

        Parameters:
        - S: spectrum of a signal
        """
        sb = librosa.feature.spectral_bandwidth(S=S, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        sb = np.mean(sb, axis=1)
        return sb

    def _get_spectral_contrast(self, S):
        """
        Returns the spectral contrast of the audio frame from its spectrum

        Parameters:
        - S: spectrum of a signal
        """
        sc = librosa.feature.spectral_contrast(S=S, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        return np.mean(sc, axis=1)

    def _get_spectral_flatness(self, S):
        """
        Returns the spectral flatness of the audio frame from its spectrum

        Parameters:
        - S: spectrum of a signal
        """
        sf = librosa.feature.spectral_flatness(S=S, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        return np.mean(sf, axis=1)

    def _get_spectral_rolloff(self, S):
        """
        Returns the spectral rolloff of the audio frame from its spectrum

        Parameters:
        - S: spectrum of a signal
        """
        sr = librosa.feature.spectral_rolloff(S=S, sr=self._sample_rate, n_fft=self._nfft, hop_length=self._hop_length, win_length=self._window_size, window=self._window_type)
        return np.mean(sr, axis=1)
    
    def _normalize(self, array):
        """
        Implements different normalizations based on the current selected normalization type

        Parameters:
        - array: array to normalize

        Returns -> normalized array (or the array as is if normalization is set to NONE)
        """
        if u.NORM_TYPE == u.Normalizations.NONE:
            return array
        
        if u.NORM_TYPE == u.Normalizations.PEAK:
            return array / np.max(np.abs(array))
        
        if u.NORM_TYPE == u.Normalizations.RMS:
            return np.sqrt(np.mean(np.square(array)))
        
        if u.NORM_TYPE == u.Normalizations.Z_SCORE:
            return (array - np.mean(array)) / np.std(array)
        
        if u.NORM_TYPE == u.Normalizations.MIN_MAX:
            min_value = np.amin(array)
            max_value = np.amax(array)
            return (array - min_value) / (max_value - min_value)

class DefaultAudioProcessor(AudioProcessor):
    """
    Implements the default chain used to process low level features
    """

    def __init__(self, sample_rate, chunk_size, np_format, nfft, hop_length, window_size, window_type, pitch_threshold):
        super().__init__(sample_rate, chunk_size, np_format, nfft, hop_length, window_size, window_type, pitch_threshold)

    def process(self, frame, inst: u.Instruments):
        """
        Processes the given audio frame according to a defined chain

        Parameters:
        - frame: chunk of audio to process

        Returns -> Low Level Features as floats ordered inside an array
        """
        frame = self._normalize(array=frame)
        S = self._compute_stft(frame)
        freqs = np.round(self._get_poly_frequencies(S, inst))
        spec_centroid = np.round(self._get_spectral_centroid(S)[0])
        spec_bandwidth = np.round(self._get_spectral_bandwidth(S)[0])
        spec_flatness = self._get_spectral_flatness(S)[0]
        spec_rolloff = np.round(self._get_spectral_rolloff(S)[0])

        self._debugger.print_data(spec_centroid)

        return [spec_centroid, spec_bandwidth, spec_flatness, spec_rolloff, *freqs[:4]]

class InputHandler(ABC):
    """
    The InputHandler abstract class declares a set of methods for processing data
    """

    def __init__(self, channel, sample_rate, chunk_size, np_sample_format, instrument: u.Instruments):
        """
        Constructor
        """
        self._input_data = None
        self._output_data = None
        self._connection_handler = OSCConnectionHandler.get_instance(address=u.NET_ADDRESS, port=u.NET_PORT)
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._np_sample_format = np_sample_format
        self.__instrument = instrument
        self._lock = mp.Lock()
        self.channel = channel

    @abstractmethod
    def process_input(self, input, inst: u.Instruments):
        """
        Abstract method to process read input data
        """
        pass

    @abstractmethod
    def send_output(self, output, inst: u.Instruments):
        """
        Abstract method to send the output of the processing 
        """
        pass

    @abstractmethod
    def process(self, data):
        """
        Abstract method to implement the processing chain
        """
        pass

    def set_instrument(self, instrument: u.Instruments):
        """
        Thread-Safe setter for the "instrument" attribute
        """
        self._lock.acquire()
        self.__instrument = instrument
        self._lock.release()

    def get_instrument(self):
        """
        Thread-Safe getter for the "instrument" attribute
        """
        self._lock.acquire()
        inst = self.__instrument
        self._lock.release()
        return inst

class LFAudioInputHandler(InputHandler):
    """
    Handles the Low Level Feature Processing
    """

    def __init__(self, channel, sample_rate, chunk_size, np_sample_format, instrument: u.Instruments):
        """
        Constructor
        """
        super().__init__(channel=channel, sample_rate=sample_rate, chunk_size=chunk_size, np_sample_format=np_sample_format, instrument=instrument)
        self._audio_processor = DefaultAudioProcessor(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            np_format=np_sample_format,
            nfft=u.DEFAULT_NFFT,
            hop_length=u.HOP_LENGTH,
            window_size = u.WINDOW_SIZE,
            window_type = u.WINDOW_TYPE,
            pitch_threshold=u.PITCH_THRESHOLD
        )

    def __no_signal(self, data):
        """
        Checks if there's an actual signal inside the read data by computing the rms and evaulating it against a predefined threshold
        """
        rms = np.sqrt(np.mean(np.square(data)))
        return rms <= u.CHANNEL_AUDIO_THRESHOLD

    def process_input(self, input, inst: u.Instruments):
        """
        Process a chunk of audio
        """
        return self._audio_processor.process(input, inst)

    def send_output(self, output, inst: u.Instruments):
        """
        Sends the output of the audio processing
        """
        msg = LFAudioMessage(data=output, channel=self.channel, instrument=inst)
        self._connection_handler.send_message(message=msg)
    
    def process(self, data):
        """
        Processes a chunk of audio or returns if there is no actual signal

        Parameters:
        - data: chunk of audio to process
        """
        if self.__no_signal(data): return

        inst = self.get_instrument()

        out = self.process_input(data, inst)
        self.send_output(out, inst)

class HFAudioInputHandler(InputHandler):
    """
    Handles the High Level Features Processing
    """

    def __init__(self, channel, sample_rate, chunk_size, np_sample_format):
        super().__init__(channel=channel, sample_rate=sample_rate, chunk_size=chunk_size, np_sample_format=np_sample_format, instrument=None)

    def process_input(self, input, inst: u.Instruments):
        """
        self.__lock.acquire()
        data_to_process = self._input_data.copy()
        self._input_data = []
        self.__lock.release()
        wf = wave.open("output.wav", 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self._sample_rate)
        wf.writeframes(b''.join(data_to_process))
        wf.close()
        """
        time.sleep(3)
        return "happy"
        
    def send_output(self, output, inst: u.Instruments):
        """
        Sends the output of the audio processing
        """
        msg = HFAudioMessage(data=output, channel=self.channel)
        self._connection_handler.send_message(msg)
    
    def process(self, data):
        """
        Processes a chunk of audio for High Level Features

        Parameters:
        - data: chunk of audio to process
        """
        out = self.process_input(data, u.Instruments.DEFAULT)
        self.send_output(out, u.Instruments.DEFAULT)
