import pandas as pd
import glob
import os
import tensorflow as tf
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Concatenate
from keras.layers import Bidirectional, LSTM, Dropout, Dense, ZeroPadding1D


# DEFINE VARIABLES
OUTPUT_DIR = '/kaggle/input/wav-16bit/train/'
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Define the strategy to use
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# DEFINE FUNCTIONS


def extract_input_target():
    path_arousal = '/kaggle/input/csv-mae/arousal_cont_average.csv'
    path_valence = '/kaggle/input/csv-mae/valence_cont_average.csv'

    data_arousal = pd.read_csv(path_arousal)
    data_valence = pd.read_csv(path_valence)
    arousal = data_arousal.drop(columns=['song_id', 'sample_45000ms'])
    valence = data_valence.drop(columns=['song_id', 'sample_45000ms'])
    path = glob.glob(os.path.join(OUTPUT_DIR, '*.wav'))
    final_path = sorted(path, key=lambda x: (int(x.split(
        '/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[-1].split('.')[0])))
    arousal = arousal.values.flatten()
    valence = valence.values.flatten()

    labels = list(zip(arousal, valence))
    return final_path, labels


def read_audio_files(filename, labels):
    audio_path = tf.io.read_file(filename)
    audio, _ = tf.audio.decode_wav(
        audio_path, desired_channels=1, desired_samples=44100)
    return audio, labels


def get_dataset(path, labels):
    file_path_ds = tf.data.Dataset.from_tensor_slices(path)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((file_path_ds, label_ds))


def load_audio(file_path, label):
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio,
                                   desired_channels=-1,
                                   desired_samples=-1)
    return audio, label


def prepare_for_training(ds, shuffle_buffer_size=1024, batch_size=64):
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Load and decode audio from file paths
    ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    # Prepare batches
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    # Prefetch
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def create_model():
    with strategy.scope():
        # Define input shape
        input_shape = (22050, 1)

        # Define input layer
        input_layer = Input(shape=input_shape)

        # Define fine-view CNN
        fine_view = Conv1D(filters=8, kernel_size=32,
                           strides=8, activation='relu')(input_layer)
        fine_view = BatchNormalization()(fine_view)
        fine_view = MaxPooling1D(pool_size=8)(fine_view)

        # Define coarse-view CNN
        coarse_view = Conv1D(filters=8, kernel_size=128,
                             strides=32, activation='relu')(input_layer)
        coarse_view = BatchNormalization()(coarse_view)
        coarse_view = MaxPooling1D(pool_size=2)(coarse_view)

        # Pad the coarse-view CNN to match the number of time steps in the fine-view CNN
        coarse_view = ZeroPadding1D(padding=((0, 1)))(coarse_view)

        # Merge the two CNNs
        merged = Concatenate(axis=-1)([fine_view, coarse_view])

        # Define Bidirectional LSTM layers
        lstm = Dropout(rate=0.5)(merged)
        lstm = Bidirectional(LSTM(units=32, return_sequences=True))(lstm)
        lstm = Bidirectional(LSTM(units=32))(lstm)
        lstm = Dropout(rate=0.5)(lstm)

        # Define output layer
        output_layer = Dense(units=2, activation='tanh')(lstm)

        # Define the final model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse', metrics=[
                      'mse', tf.keras.metrics.RootMeanSquaredError()], run_eagerly=True)
        model.summary()
    return model
