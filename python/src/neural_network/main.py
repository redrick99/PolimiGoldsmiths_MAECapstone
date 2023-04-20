#%%
import tensorflow as tf
import numpy as np
from keras.utils import plot_model
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Concatenate
from keras.layers import Bidirectional, LSTM, Dropout, Dense, ZeroPadding1D
from keras.callbacks import ReduceLROnPlateau
import glob

import os
import training

print("DONE IMPORT")

################################################### NEURAL NETWORK ###################################################
#%% Build th Neural Network

# Define input shape
input_shape = (22050,1)

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
# plot_model(model, to_file='FirstNN.png', show_shapes=True, show_layer_names=True)


####################################################### FIT #######################################################
#%% Model Fit

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
train_files = glob.glob(os.path.join(training.OUTPUT_DIR_TRAIN, '*.pkl'))
x_tr, y_v, y_a = training.get_data(train_files)
test_files = glob.glob(os.path.join(training.OUTPUT_DIR_TEST, '*.pkl'))
x_te, y_v_te, y_a_te = training.get_data(test_files)
#%%
# if the accuracy does not increase over 10 epochs, reduce the learning rate by half.
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=0.00005, verbose=1)
batch_size = 128 ## 128 samples each time



#%%
y_tr = np.column_stack((y_v, y_a))
y_te = np.column_stack((y_v_te, y_a_te))

#%%
model.fit(x=x_tr,
  y=y_tr,
  batch_size=batch_size,
  epochs=50,
  verbose=1,
  shuffle=True,
  validation_data=(x_te, y_te),
  callbacks=[reduce_lr])
model.save("model.h5")

# %%
