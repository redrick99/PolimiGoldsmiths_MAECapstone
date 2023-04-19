#%%
import librosa
import os
import numpy as np
import pickle
from glob import iglob
import pandas as pd

DATA_AUDIO_DIR = '../../../Music_mp3/clips_45seconds'
TARGET_SR = 44100
AUDIO_LENGTH = 22050
OUTPUT_DIR = './output'
OUTPUT_DIR_TRAIN = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_DIR_TEST = os.path.join(OUTPUT_DIR, 'test')

#%%
def read_audio_from_filename(filename):
    audio = [[0 for j in range(22050)] for i in range(60)]
    for i in range(60):
        line = librosa.load(filename,
                                    offset=15.0+i/2, duration=0.5, sr=TARGET_SR)
        audio[i] = list(line)
        audio[i] = (librosa.util.normalize(audio[i][0]), TARGET_SR)

    # audio is a matrix composed by 22050 columns and 60 rows. Eaxh row contains the samples of 0.5 seconds
    # of the track from 15.0s to 45.0s
    return audio

# %%

#%%
def convert_data():
    out = {}
    path, arousal, valance = extract_input_target()
    for i, (x_i, a_i, v_i) in enumerate(zip(path, arousal, valance)):
        audio_buf = read_audio_from_filename(os.path.join(DATA_AUDIO_DIR,x_i))
        for k, (audio_sample, audio_valance, audio_arousal) in enumerate(zip(audio_buf, a_i, v_i)):
            if len(audio_sample[0]) < AUDIO_LENGTH:
                print(len(audio_sample[0]))
                audio_sample[0] = np.concatenate((audio_sample[0], np.zeros(shape=(AUDIO_LENGTH - len(audio_sample[0])))))
                print('PAD New length =', len(audio_sample[0]))
            out[k] = {
                'audio': audio_sample[0],
                'sr': TARGET_SR,
                'valence': audio_valance,
                'arousal': audio_arousal
                }

        if i // 744*0.3 == 0:
            output_folder = OUTPUT_DIR_TEST
        else:
            output_folder = OUTPUT_DIR_TRAIN

        for j in range(60):
            print(audio_sample)
            output_filename = os.path.join(output_folder, str(i) + str('_') + str(j) + '.pkl')
            with open(output_filename, 'wb') as w:
                pickle.dump(out[j], w)


# %%
# convert_data()

#%% TEST
def extract_input_target():
    path_arousal = '../../../Music_mp3/annotations/arousal_cont_average.csv'
    path_valence = '../../../Music_mp3/annotations/valence_cont_average.csv'
    path_info = '../../../Music_mp3/annotations/songs_info.csv'

    data_arousal = pd.read_csv(path_arousal)
    data_valence = pd.read_csv(path_valence)
    data_info = pd.read_csv(path_info)

    path = data_arousal['song_id'].apply(lambda x: str(x) + '.mp3')

    arousal = data_arousal['sample_'+str(150)+'00ms']

    arousal = np.zeros(((744, 60)))
    valance = np.zeros((744, 60))

    for j in range(744):
        for i in range(60):
            arousal[j][i] = data_arousal['sample_'+str(150+i*5)+'00ms'][j]
            valance[j][i] = data_valence['sample_'+str(150+i*5)+'00ms'][j]

    return path, arousal, valance

#%%
i, a, v = extract_input_target()

# %%
audio = read_audio_from_filename(os.path.join(DATA_AUDIO_DIR,i[8]))

# %%
def get_data(file_list):
    def load_into(_filename, _x, _y):
        with open(_filename, 'rb') as f:
            audio_element = pickle.load(f)
            _x.append(audio_element['audio'])
            _y.append(int(audio_element['class_id']))

    x, y = [], []
    for filename in file_list:
        load_into(filename, x, y)
    return np.array(x), np.array(y)