#%%
import tensorflow as tf
import numpy as np
print("ONE")
from keras.utils import plot_model
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Concatenate
from keras.layers import Bidirectional, LSTM, Dropout, Dense, ZeroPadding1D
from keras.callbacks import ReduceLROnPlateau
print('keras')
import glob
import sklearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
print('sklearn')


import os
print('OS')
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
              loss='mse')
# print(model.summary())
train_files = glob.glob(os.path.join(training.OUTPUT_DIR_TRAIN, '*.pkl'))

X, y_v, y_a = training.get_data(train_files)
y = np.column_stack((y_v, y_a))
kf = KFold(n_splits=10)
rms = []
rms_test = []
X, X_remain, y_train, y_remain = train_test_split(X, y, train_size=0.9)
for train_index, validation_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", validation_index)
    X_train, X_validation = X[train_index], X[validation_index]
    y_train, y_validation = y[train_index], y[validation_index]
    # tf.convert_to_tensor(X_train, dtype=tf.float32)
    # tf.convert_to_tensor(y_train, dtype=tf.float32)
    # Split the data into training, validation, and test sets with 8:1:1 ratio
    model.fit(x=X_train,
              y=y_train,
              batch_size=128,
              epochs=10,
              verbose=1,
              shuffle=True)
    model.save("model.h5")
    y_val_pred = model.predict(X_validation)
    rms.append(np.sqrt(mean_squared_error(y_validation, y_val_pred)))
    y_test_predict = model.predict(X_remain)
    rms_test.append(np.sqrt(mean_squared_error(y_remain, y_test_predict)))
    
    
#%%
