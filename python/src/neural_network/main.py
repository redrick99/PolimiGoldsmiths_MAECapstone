# %%
# IMPORT LIBRARY
import tensorflow as tf
import functions_train as ft
import numpy as np
from os import path
from tensorflow.keras.callbacks import EarlyStopping, train_test_split
from sklearn.model_selection import KFold
import training


# Load meta.csv containing file-paths and labels as pd.DataFrame
# %%
path, labels = ft.extract_input_target()
X_train, X_test, y_train, y_test = train_test_split(
    path, labels, test_size=0.1, random_state=42)
ds_train = ft.get_dataset(X_train, y_train)
ds_test = ft.get_dataset(X_test, y_test)


####################################################### FIT #######################################################
# Definizione del numero di fold della cross-validation
# %%
num_folds = 10
batch_size = 1024
len_dataset = len(list(ds_train.as_numpy_iterator()))*0.8
fold_size = len_dataset//num_folds
kf = KFold(n_splits=num_folds, shuffle=True)
best_rmse = 1
for fold, (train_indexes, val_indexes) in enumerate(kf.split(X_train)):
    print('Starting Fold:', fold+1)
    start, end = fold * fold_size, (fold + 1) * fold_size

    # Definizione degli insiemi di training e di validation per questo fold
    val_data = ds_train.take(len(val_indexes))
    train_data = ds_train.skip(len(val_indexes))
    print("split dataset----")
    train_steps = int(train_data.cardinality().numpy()//batch_size)
    print(train_steps)
    val_steps = int(val_data.cardinality().numpy()//batch_size)
    print(val_steps)

    train_data = ft.prepare_for_training(train_data)
    val_data = ft.prepare_for_training(val_data)
    print("prepare for training")
    # Creazione e addestramento del modello per questo fold
    model = ft.create_model()
    early_stop = EarlyStopping(
        monitor='val_loss',  # monitora la loss sul set di validazione
        patience=10,  # numero di epoche consecutive senza miglioramento
        mode='min',  # minimizza la loss
        verbose=1  # stampa messaggi durante l'addestramento
    )
    model.fit(train_data, epochs=1000, validation_data=val_data, verbose=1,
              validation_steps=val_steps, steps_per_epoch=train_steps, callbacks=[early_stop])
    print('Evaluate------')
    loss, mse, rmse = model.evaluate(train_data, steps=train_steps)
    print('LOSS:', loss)
    print('mse:', mse)
    print('rmse:', rmse)
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model


# PREDICTION
# %%
model = tf.keras.models.load_model('./output/modello.h5')
audio_test = training.librosa.load(
    './output/train/808_38.wav', sr=training.TARGET_SR)
prediction = model.predict(np.expand_dims(audio_test[0], axis=0))
# %%
