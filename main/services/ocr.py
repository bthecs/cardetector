import os
import random
import string
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.activations import softmax
from custom import cce, cat_acc, plate_acc, top_3_k

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
        self.model = tf.keras.models.load_model('model.h5', custom_objects=self.custom_objects)
    
    
    
    @tf.function
    def predict_array(self, img):
        
        predictions = self.model(img, training=False)
        return predictions
    
    def plate_probs(self, prediction):
        prediction = prediction.reshape((7,37))
        probs = np.max(prediction, axis=1)
        prediction = np.argmax(prediction, axis=-1)
        plate = list(map(lambda x: self.alphabet[x], prediction))
        return probs, plate
    
    def process_roi(self, image):
        
        #cambiar imagen a escala de grises
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(140,70), interpolation=cv2.INTER_LINEAR)
        img = img[np.newaxis, ..., np.newaxis] / 255.
        plate = tf.constant(img, dtype=tf.float32)
        prediction = self.predict_array(plate).numpy()
        plate_probs, plate = self.plate_probs(prediction)
        print(plate)
        plate_final = ''.join(plate).strip('_')
        print(f"Plate: {plate_final}")
        print(f"Probs: {plate_probs}")
        
        return plate_final
    

    def main(self, image):        
        self.process_roi(image)

