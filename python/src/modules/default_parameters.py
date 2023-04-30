from modules.utilities import Normalizations

# Ummutable constants (defined on startup)
NET_PARAMETERS = {
    'outNetAddress' : "127.0.0.1",
    'outNetPort' : 12345,
    'inNetAddress' : "127.0.0.1",
    'inNetPort' : 1337,
}

OSC_MESSAGES_PARAMETERS = {
    'outLf' : "/LFmsg_ch",
    'outHf' : "/HFmsg_ch",

    'inStart' : "/START",
    'inStop' : "/STOP",
    'inChannelSettings' : "/ch_settings"
}

AUDIO_PROCESSING_PARAMETERS = {
    'chunkSize' : 4096,
    'signalThreshold' : 0.005,
    'nfft' : 4096,
    'hopLength' : int(4096/2),
    'winSize' : 4096,
    'winType' : 'hann',
    'hfNumberOfSamples' : 22050,
    'pitchThreshold' : 0.2,
    'normType' : Normalizations.PEAK
}

CPU_PARAMETERS = {
    'numLfCores' : 2,
    'numHfCores' : 1
}