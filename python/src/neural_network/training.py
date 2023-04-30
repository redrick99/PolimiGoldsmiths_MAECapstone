# %%
import wave
import librosa
import os
import numpy as np
import pickle
import glob
from glob import iglob
import pandas as pd
import soundfile as sf

DATA_AUDIO_DIR = '../../../Music_mp3/clips_45seconds'
TARGET_SR = 44100
AUDIO_LENGTH = 22050
OUTPUT_DIR = './output'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')

# %%


def read_audio_from_filename(filename):
    audio = [[0 for j in range(22050)] for i in range(60)]
    for i in range(60):
        line = librosa.load(filename,
                            offset=15.0+i/2, duration=0.5, sr=TARGET_SR)
        audio[i] = list(line)
        audio[i] = (librosa.util.normalize(audio[i][0]), TARGET_SR)
    # audio is a matrix composed by 22050 columns and 60 rows. Each row contains the samples of 0.5 seconds
    # of the track from 15.0s to 45.0s
    return audio


# %%
def convert_data():
    path, _, _ = extract_input_target()
    for x_i in path:
        print(x_i)
        audio_buf = read_audio_from_filename(
            os.path.join(DATA_AUDIO_DIR, (x_i+'.mp3')))
        for k, (audio_sample) in enumerate(audio_buf):
            # Zero padding if the sample is short)
            if len(audio_sample[0]) < AUDIO_LENGTH:
                # print(audio_sample[0])
                audio_sample = (np.concatenate((audio_sample[0], np.zeros(
                    shape=(AUDIO_LENGTH - len(audio_sample[0]))))), TARGET_SR)

            output_folder = OUTPUT_DIR_TRAIN
            output_filename = os.path.join(
                output_folder, str(x_i) + str('_') + str(k)+'.wav')
            sf.write(output_filename,
                     audio_sample[0], TARGET_SR, subtype='PCM_16')


# %%
def extract_input_target():
    path_arousal = '../../../Music_mp3/annotations/arousal_cont_average.csv'
    path_valence = '../../../Music_mp3/annotations/valence_cont_average.csv'

    data_arousal = pd.read_csv(path_arousal)
    data_valence = pd.read_csv(path_valence)

    path = data_arousal['song_id'].apply(lambda x: str(x))

    arousal = data_arousal['sample_'+str(150)+'00ms']

    arousal = np.zeros(((744, 60)))
    valance = np.zeros((744, 60))

    for j in range(744):
        for i in range(60):
            arousal[j][i] = data_arousal['sample_'+str(150+i*5)+'00ms'][j]
            valance[j][i] = data_valence['sample_'+str(150+i*5)+'00ms'][j]

    return path, arousal, valance


# %%
