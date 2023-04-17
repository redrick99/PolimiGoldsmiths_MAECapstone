import librosa
import os
import numpy as np
import pickle
from glob import iglob
import pandas as pd

DATA_AUDIO_DIR = '/Users/francescopiferi/PycharmProjects/Music_mp3/clips_45seconds'
TARGET_SR = 44100
AUDIO_LENGTH = 22050
OUTPUT_DIR = '/Users/francescopiferi/PycharmProjects/NeuralNetwork/output'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')

#%%
def read_audio_from_filename(filename):
    audio = [[0 for j in range(22050)] for i in range(60)]
    for i in range(60):
        line = librosa.load('/Users/francescopiferi/PycharmProjects/Music_mp3/clips_45seconds/2.mp3',
                                    offset=15.0+i/2, duration=0.5, sr=TARGET_SR)
        audio[i] = line

    # audio is a matrix composed by 22050 columns and 60 rows. Eaxh row contains the samples of 0.5 seconds
    # of the track from 15.0s to 45.0s
    return audio

#%%
def convert_data():
    for i,(x_i,t_i) in enumerate(zip(extract_input_target())):
        class_id = t_i;
        audio_buf, target_sr = read_audio_from_filename(os.path.join(DATA_AUDIO_DIR,x_i), target_sr=TARGET_SR)
        # normalize mean 0, variance 1
        audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
        original_length = len(audio_buf)
        print(i, os.path.join(DATA_AUDIO_DIR,x_i), original_length, np.round(np.mean(audio_buf), 4), np.std(audio_buf))
        if original_length < AUDIO_LENGTH:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
            print('PAD New length =', len(audio_buf))
        elif original_length > AUDIO_LENGTH:
            audio_buf = audio_buf[0:AUDIO_LENGTH]
            print('CUT New length =', len(audio_buf))


        if i // 744*0.3 == 0:
            output_folder = OUTPUT_DIR_TEST
        else:
            output_folder = OUTPUT_DIR_TRAIN

        for j in len(audio_buf):
            output_filename = os.path.join(output_folder, str(i) +str('_') + str(j%100) + str(j%10) + str(j) + '.pkl')

        out = {'class_id': class_id,
               'audio': audio_buf,
               'sr': TARGET_SR}
        with open(output_filename, 'wb') as w:
             pickle.dump(out, w)


#%% TEST
def extract_input_target():
    path_arousal = '/Users/francescopiferi/PycharmProjects/Music_mp3/annotations/arousal_cont_average.csv'
    path_valence = '/Users/francescopiferi/PycharmProjects/Music_mp3/annotations/valence_cont_average.csv'
    path_info = '/Users/francescopiferi/PycharmProjects/Music_mp3/annotations/songs_info.csv'

    data_arousal = pd.read_csv(path_arousal)
    data_valence = pd.read_csv(path_valence)
    data_info = pd.read_csv(path_info)

    target_arousal = data_arousal['song_id'].apply(lambda x: str(x) + '.mp3')
    target_valence = data_valence['song_id'].apply(lambda x: str(x) + '.mp3')
    path = data_arousal['song_id'].apply(lambda x: str(x) + '.mp3')

    # Check if the arrays are equal
    """"
    if len(target_arousal) != len(target_valence):
        print("a and b are not equal")
    else:
        for i in range(len(target_arousal)):
            if target_arousal[i] != target_valence[i]:
                print("a and b are not equal")
                break
        print("a and b are equal")
    """

    v_arousal = data_arousal['sample_15000ms'][0]


    # values = [(v_a,v_v), (v_a,v_v)]


    return path, v_arousal

#%%
i, v = extract_input_target()
