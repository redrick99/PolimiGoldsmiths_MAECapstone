#%%
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Concatenate
from keras.layers import Bidirectional, LSTM, Dropout, Dense, ZeroPadding1D
import training

print("DONE IMPORT")

#%% Load the audio file
audio_path = '/Users/francescopiferi/Desktop/Beatles_LetItBe.wav'
y, sr = librosa.load(audio_path)
y = y[:15*sr]
print("DONE LOAD AND CUT")

#%% Compute the spectrogram
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
spectrogram = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
print("DONE COMPUTATION SPECTROGRAM")

#%% Normalization
spectrogram -= np.mean(spectrogram)
spectrogram /= np.std(spectrogram)
print("DONE NORMALIZATION")

#%% Prepare for the print and Print
input = librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.xlabel('Time')
plt.ylabel('Frequency')
# plt.title('Spectrogram Beatles Image NORMALZIED')
plt.colorbar(format='%+2.0f dB')
plt.show()

print("DONE PRINT")

#%% NO
spectrogram = np.expand_dims(spectrogram, axis=0)

#%% Flatten the spectrogram
spectrogram_1d = spectrogram.reshape(-1,1) # From a  matrix return a (array, 1) matrix
print("DONE FLATTEN")

################################################### NEURAL NETWORK ###################################################
#%% Build th Neural Network

# Define input shape
input_shape = spectrogram_1d.shape

# Define input layer
input_layer = Input(shape=input_shape)

# Define fine-view CNN
fine_view = Conv1D(filters=8, kernel_size=32, strides=8, activation='relu')(input_layer)
fine_view = BatchNormalization()(fine_view)
fine_view = MaxPooling1D(pool_size=8)(fine_view)

# Define coarse-view CNN
coarse_view = Conv1D(filters=8, kernel_size=128, strides=32, activation='relu')(input_layer)
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

#%% Plot Neural Network
plot_model(model, to_file='FirstNN.png', show_shapes=True, show_layer_names=True)

#%%
v = training.extract_input_target()

####################################################### FIT #######################################################
#%%
""""
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
train_files = glob(os.path.join(OUTPUT_DIR_TRAIN, '.pkl'))
x_tr, y_tr = get_data(train_files)
y_tr = to_categorical(y_tr, num_classes=num_classes)
test_files = glob(os.path.join(OUTPUT_DIR_TEST, '.pkl'))
x_te, y_te = get_data(test_files)
y_te = to_categorical(y_te, num_classes=num_classes)
print('x_tr.shape =', x_tr.shape)
print('y_tr.shape =', y_tr.shape)
print('x_te.shape =', x_te.shape)
print('y_te.shape =', y_te.shape)
model = keras.models.load_model("/content/model.h5")
K.set_value(model.optimizer.learning_rate, 0.000005)
# if the accuracy does not increase over 10 epochs, reduce the learning rate by half.
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=0.00005, verbose=1)
batch_size = 128
model
model.fit(x=x_tr,
  y=y_tr,
  batch_size=batch_size,
  epochs=50,
  verbose=1,
  shuffle=True,
  validation_data=(x_te, y_te),
  callbacks=[reduce_lr])
model.save("model.h5")
"""
