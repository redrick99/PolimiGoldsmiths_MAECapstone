# %%

# This script is used to convert the audio in frames of 0.5 seconds
# with a total length of 45 seconds

import librosa
import os
import numpy as np
import pandas as pd
import soundfile as sf

# CHOOSE THE DATA AUDIO DIRECTION
DATA_AUDIO_DIR = '../../../Music_mp3/DEAM_audio/MEMD_audio'
TARGET_SR = 44100
AUDIO_LENGTH = 22050
OUTPUT_DIR = './output'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')

# %%

# shift by default is set to 0, by changing it the pitch shift is applied
def read_audio_from_filename(filename, shift = 0):
    audio = []
    for i in range(60):
        line = librosa.load(filename,
                            offset=15.0+i/2, duration=0.5, sr=TARGET_SR)
        line = librosa.effects.pitch_shift(line[0], sr = TARGET_SR, n_steps = shift)
        audio.append(librosa.util.normalize(line))
    # audio is a matrix composed by 22050 columns and 60 rows. Each row contains the samples of 0.5 seconds
    # of the track from 15.0s to 45.0s
    return audio


# %%
def convert_data():
    path = extract_input_target()
    for x_i in path:
        print(x_i)
        audio_buf = read_audio_from_filename(
            os.path.join(DATA_AUDIO_DIR, (x_i+'.mp3')))
        # PITCH SHIFT FOR DATA AUGUMENTATION
        # audio_buf_shift = read_audio_from_filename(
        # os.path.join(DATA_AUDIO_DIR, (x_i+'.mp3')), shift=random.choice([-1,1]))
        for k, (audio_sample) in enumerate(audio_buf):
        
        # PITCH SHIFT FOR DATA AUGUMENTATION
        # for k, (audio_sample, audio_sample_shift) in enumerate(zip(audio_buf, audio_buf_shift)):
            
            # Zero padding if the sample is short)

            if len(audio_sample) < AUDIO_LENGTH:
                audio_sample = np.concatenate((audio_sample, np.zeros(
                    shape=(AUDIO_LENGTH - len(audio_sample)))))
            
            # PITCH SHIFT FOR DATA AUGUMENTATION
            # if len(audio_sample_shift) < AUDIO_LENGTH:
            # audio_sample_shift = np.concatenate((audio_sample_shift, np.zeros(
            # shape=(AUDIO_LENGTH - len(audio_sample_shift)))))

            output_folder = OUTPUT_DIR_TRAIN
            output_filename = os.path.join(
                output_folder, str(x_i) + str('_') + str(k)+'.wav')
            sf.write(output_filename,
                     audio_sample, TARGET_SR, subtype='PCM_16')
            
            # PITCH SHIFT FOR DATA AUGUMENTATION

            # output_filename = os.path.join(
            # output_folder, str(int(x_i)+1000) + str('_') + str(k)+'.wav')
            # sf.write(output_filename,
            #          audio_sample_shift, TARGET_SR, subtype='PCM_16')
            
# %%
def extract_input_target():
    path_arousal = r'../../../Music_mp3/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv'

    data_arousal = pd.read_csv(path_arousal)

    path = data_arousal['song_id'].apply(lambda x: str(x))

    return path


# %%
convert_data()
# %%
