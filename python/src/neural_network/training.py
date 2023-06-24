# This script is used to convert the audio in frames of 0.5 seconds
# with a total length of 45 seconds

import librosa
import os
import numpy as np
import pandas as pd
import soundfile as sf
import random

DATA_AUDIO_DIR = '../../../Music_mp3/DEAM_audio/MEMD_audio'
TARGET_SR = 44100
AUDIO_LENGTH = 22050
OUTPUT_DIR = './output'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')


def read_audio_from_filename(filename, shift = 0):
    """Read audio from the specified path and .

    **Args:**

    `filename`: Path of the audio to read.

    `shift`: Argument used only if we want perform data augumentation. Shift the song with the selected number of semitones.

    **Returns:**
    Audio is a matrix composed by 22050 columns and 60 rows. Each row contains the samples of 0.5 seconds
    of the track from 15.0s to 45.0s.

    """
    audio = []
    for i in range(60):
        line = librosa.load(filename,
                            offset=15.0+i/2, duration=0.5, sr=TARGET_SR)
        line = librosa.effects.pitch_shift(line[0], sr = TARGET_SR, n_steps = shift)
        audio.append(librosa.util.normalize(line))
    return audio

def convert_data(data_augumentation = 0):
    """Reads audio from the specified path and converts it to WAV format in PCM 16 bit format.

    **Args:**

    `data_augmentation`: Flag indicating whether to perform data augmentation. Default is 0 (no data augmentation).
    """

    path = extract_input_target()
    for x_i in path:
        print(x_i)
        audio_buf = read_audio_from_filename(
            os.path.join(DATA_AUDIO_DIR, (x_i+'.mp3')))
        
        # With Data augumentation
        if(data_augumentation == 1):
            audio_buf_shift = read_audio_from_filename(
            os.path.join(DATA_AUDIO_DIR, (x_i+'.mp3')), shift=random.choice([-1,1]))
            for k, (audio_sample, audio_sample_shift) in enumerate(zip(audio_buf, audio_buf_shift)):
            
                # Zero padding if the sample is short)

                if len(audio_sample) < AUDIO_LENGTH:
                    audio_sample = np.concatenate((audio_sample, np.zeros(
                    shape=(AUDIO_LENGTH - len(audio_sample)))))
            
                if len(audio_sample_shift) < AUDIO_LENGTH:
                    audio_sample_shift = np.concatenate((audio_sample_shift, np.zeros(
                    shape=(AUDIO_LENGTH - len(audio_sample_shift)))))

                output_folder = OUTPUT_DIR_TRAIN
                output_filename = os.path.join(
                output_folder, str(x_i) + str('_') + str(k)+'.wav')
                sf.write(output_filename,
                     audio_sample, TARGET_SR, subtype='PCM_16')

                output_filename = os.path.join(
                output_folder, str(int(x_i)+1000) + str('_') + str(k)+'.wav')
                sf.write(output_filename,
                    audio_sample_shift, TARGET_SR, subtype='PCM_16')

        for k, (audio_sample) in enumerate(audio_buf):
            
            # Zero padding if the sample is short

            if len(audio_sample) < AUDIO_LENGTH:
                audio_sample = np.concatenate((audio_sample, np.zeros(
                    shape=(AUDIO_LENGTH - len(audio_sample)))))
            
            output_folder = OUTPUT_DIR_TRAIN
            output_filename = os.path.join(
                output_folder, str(x_i) + str('_') + str(k)+'.wav')
            sf.write(output_filename,
                     audio_sample, TARGET_SR, subtype='PCM_16')
        
def extract_input_target():
    """Extract the path of each songs.

    **Returns:**

    A list of song paths
    """
    path_arousal = r'../../../Music_mp3/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv'

    data_arousal = pd.read_csv(path_arousal)

    path = data_arousal['song_id'].apply(lambda x: str(x))

    return path
