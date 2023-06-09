import os
import random
import string
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.activations import softmax
from .custom import cce, cat_acc, plate_acc, top_3_k

class DetectOCR:

    def __init__(self):
        self.alphabet = string.digits + string.ascii_uppercase + '_'
        self.custom_objects = {
            'cce': cce,
            'cat_acc': cat_acc,
            'plate_acc': plate_acc,
            'top_3_k': top_3_k,
            'softmax': softmax
        }
        self.model = tf.keras.models.load_model('main\services\model.h5', custom_objects=self.custom_objects)
    
    
    
    @tf.function
    def predict_array(self, img):
        # Realiza una inferencia de placas utilizando el modelo cargado y la imagen de entrada.
        predictions = self.model(img, training=False)
        return predictions
    
    def plate_probs(self, prediction):
        # Reorganiza la forma de la predicción en (7, 37) para reflejar las dimensiones de la placa.
        prediction = prediction.reshape((7,37))
        # Calcula las probabilidades máximas para cada posición de la placa.
        probs = np.max(prediction, axis=1)
        # Obtiene el índice de la letra con la probabilidad más alta para cada posición de la placa.
        prediction = np.argmax(prediction, axis=-1)
        # Mapea los índices de las letras a las letras correspondientes utilizando el atributo "alphabet".
        plate = list(map(lambda x: self.alphabet[x], prediction))
        # Devuelve las probabilidades y las letras de la placa.
        return probs, plate
    
    def process_roi(self, image):
        
        #cambiar imagen a escala de grises
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # redimensionar la imagen
        img = cv2.resize(img, dsize=(140,70), interpolation=cv2.INTER_LINEAR)
        # agregar una dimensión al array para que cumpla con el modelo
        img = img[np.newaxis, ..., np.newaxis] / 255.
        # Convierte el tensor de datos a un tensor constante de TensorFlow
        plate = tf.constant(img, dtype=tf.float32)
        # Realiza una inferencia de placas utilizando el modelo cargado y la imagen de entrada.
        prediction = self.predict_array(plate).numpy()
        # Obtiene las probabilidades de las placas en la imagen
        plate_probs, plate = self.plate_probs(prediction)
        # print(plate)
        plate_final = ''.join(plate).strip('_')        
        return plate_final
    

    def main(self, image):        
        plate_final = self.process_roi(image)
        return plate_final

