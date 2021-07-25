# Módulo con funciones para cargar y guardar los modelos predictivos en la ruta especificada
import os
import json
import time
import tensorflow as tf
from tensorflow.keras.models import model_from_json  # Cargar los modelos guardados
import tensorflow as tf


def abrir_img(path, img_height = 224, img_width=224, batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    print(train_ds.class_names)
    return train_ds, val_ds










