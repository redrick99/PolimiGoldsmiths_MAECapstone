# IMPORT LIBRARY
import tensorflow as tf
import functions_train as ft
from os import path
from tensorflow.keras.callbacks import EarlyStopping, train_test_split,ReduceLROnPlateau

# We have executed this code on Kaggle 

if __name__ =="__main__":
    
    # Load meta.csv containing file-paths and labels as pd.DataFrame

    path,labels = ft.extract_input_target()
    batch_size = 32

    # Split Dataset in training,validation and test

    X_train, X_test, y_train, y_test = train_test_split(path, labels, test_size=0.2, random_state=42)
    ds_train = ft.get_dataset(X_train,y_train)
    X_test,X_validation,y_test,y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    ds_validation = ft.get_dataset(X_validation,y_validation)
    ds_test = ft.get_dataset(X_test,y_test)
    train_data = ft.prepare_for_training(ds_train,batch_size = batch_size)
    val_data = ft.prepare_for_training(ds_validation,batch_size = batch_size)
    test_data = ft.prepare_for_training(ds_test,batch_size = batch_size)

    # Create the model

    model = ft.create_model()

    #Create callbacks

    early_stop= EarlyStopping(
        monitor='val_loss', # monitora la loss sul set di validazione
        patience=10, # numero di epoche consecutive senza miglioramento
        mode='min', # minimizza la loss
        verbose=1 # stampa messaggi durante l'addestramento
    )

    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.0001, patience=5, verbose=1, min_lr=0.0001)

    #Starting fit
    model.fit(train_data, epochs=1000,validation_data=val_data,steps_per_epoch=100,validation_steps=60, verbose='auto',callbacks=[early_stop,lr_callback])
