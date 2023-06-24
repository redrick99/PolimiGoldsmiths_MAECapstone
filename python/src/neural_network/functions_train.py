import pandas as pd
import glob
import os
import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Concatenate,AveragePooling1D
from keras.layers import Bidirectional, LSTM, Dropout, Dense, ZeroPadding1D
from tensorflow.keras.regularizers import l2


# DEFINE VARIABLES
INPUT_DIR = '/kaggle/input/wav-16bit/train/'
AUTOTUNE = tf.data.experimental.AUTOTUNE

def extract_input_target():
    """Linking the audio path with the associated valence-arousal

    **Returns:**
    The path of the audio and its corresponding labels

    """
    path_arousal = '/kaggle/input/deam-mediaeval-dataset-emotional-analysis-in-music/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv'
    path_valence = '/kaggle/input/deam-mediaeval-dataset-emotional-analysis-in-music/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv'

    data_arousal = pd.read_csv(path_arousal)
    data_valence = pd.read_csv(path_valence)
    arousal = data_arousal.drop(columns =['song_id'])
    valence = data_valence.drop(columns = ['song_id'])
    arousal = arousal.iloc[:, :60]
    valence = valence.iloc[:,:60]
    path = glob.glob(os.path.join(INPUT_DIR, '*.wav'))
    final_path= sorted(path, key=lambda x: (int(x.split('/')[-1].split('_')[0]),int(x.split('/')[-1].split('_')[-1].split('.')[0])))
    arousal = arousal.values.flatten()
    arousal = np.concatenate((arousal,arousal))
    valence = valence.values.flatten()
    valence = np.concatenate((valence,valence))
    
    labels = list(zip(arousal,valence))
    return final_path,labels

def get_dataset(path, labels):
    """Creates a dataset by zipping audio file paths and their corresponding labels.

    **Args:**
        path (List[str]): List of file paths of audio files.
        labels (List[str]): List of labels associated with the audio files.

    **Returns:**
        dataset (tf.data.Dataset): A TensorFlow dataset containing zipped file paths and labels.
    """
    file_path_ds = tf.data.Dataset.from_tensor_slices(path)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((file_path_ds, label_ds))

def load_audio(file_path, label):
    """Reads audio files and their corresponding labels.

    **Args:**
        file_path (str): File path of the audio file.
        labels (str): Label associated with the audio file.

    **Returns:**
        audio (Tensor): Audio data as a TensorFlow Tensor.
        labels (str): Corresponding labels for the audio file.
    """
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio)
    return audio, label


def prepare_for_training(ds, shuffle_buffer_size=1024, batch_size=64):

    """Prepares a dataset for training by applying transformations.

    **Args:**
        ds (tf.data.Dataset): The input dataset.
        shuffle_buffer_size (int): The buffer size for shuffling the dataset.
        batch_size (int): The batch size for creating batches of data.

    **Returns:**
        ds (tf.data.Dataset): The prepared dataset.
    """

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

def R_squared(y, y_pred):
    """Calculates the coefficient of determination (R-squared).

    **Args:**
        y (tf.Tensor): The true values.
        y_pred (tf.Tensor): The predicted values.

    **Returns:**
        r2 (tf.Tensor): The R-squared value.
    """
  
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, residual/total)
    return r2

def create_model():
    """Creates Model.

    **Returns:**
        model (tf.keras.Model): The created model.
    """

    # Define input shape
    input_shape = (22050, 1)

    # Define input layer
    input_layer = Input(shape=input_shape)


    # Define fine-view CNN
    fine_view = Conv1D(filters=20, kernel_size=256, strides=32, activation='relu', kernel_regularizer=l2(0.0001), kernel_initializer='normal')(input_layer)
    fine_view = BatchNormalization()(fine_view)
    fine_view = AveragePooling1D(pool_size=8)(fine_view)

    # Define fine-view CNN 1
    fine_view1 = Conv1D(filters=20, kernel_size=256, strides=32, activation='relu', kernel_regularizer=l2(0.0001), kernel_initializer='normal')(input_layer)
    fine_view1 = BatchNormalization()(fine_view1)
    fine_view1 = AveragePooling1D(pool_size=8)(fine_view1)

    # Define fine-view CNN 2
    fine_view2 = Conv1D(filters=20,kernel_size=512, strides=64, activation='relu', kernel_regularizer=l2(0.0001), kernel_initializer='normal')(input_layer)
    fine_view2 = BatchNormalization()(fine_view2)
    fine_view2 = AveragePooling1D(pool_size=4)(fine_view2)
    
    # Define fine-view CNN 2
    fine_view3 = Conv1D(filters=20,kernel_size=512, strides=64, activation='relu', kernel_regularizer=l2(0.0001), kernel_initializer='normal')(input_layer)
    fine_view3 = BatchNormalization()(fine_view3)
    fine_view3 = AveragePooling1D(pool_size=4)(fine_view3)
    
    #fine_view1 = ZeroPadding1D(padding=((0, 1)))(fine_view1)
    fine_view2 = ZeroPadding1D(padding=((0, 1)))(fine_view2)
    fine_view3 = ZeroPadding1D(padding=((0, 1)))(fine_view3)

    # Merge the three CNNs
    merged = Concatenate(axis=-1)([fine_view, fine_view1, fine_view2,fine_view3])

    # Define Bidirectional LSTM layers
    lstm = Dropout(rate=0.5)(merged)
    lstm = Bidirectional(LSTM(units=32, return_sequences=True))(lstm)
    lstm = Bidirectional(LSTM(units=32))(lstm)
    lstm = Dropout(rate=0.5)(lstm)
    lstm = tf.expand_dims(lstm, axis=1)
    lstm = Bidirectional(LSTM(units=32, return_sequences=True))(lstm)
    lstm = Bidirectional(LSTM(units=32))(lstm)
    lstm = Dropout(rate=0.5)(lstm)

    # Define output layer
    output_layer = Dense(units=2, activation='tanh')(lstm)

    # Define the final model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', tf.keras.metrics.RootMeanSquaredError(), 'accuracy',R_squared], run_eagerly=True)
    model.summary()
    return model
