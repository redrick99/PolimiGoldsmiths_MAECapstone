import librosa

def pitch_shift(filename):
    audio_pitch = librosa.effects.pitch_shift(y = filename, n_steps = -1)