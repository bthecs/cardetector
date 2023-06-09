from tensorflow.python.saved_model import tag_constants
import numpy as np
import cv2
import tensorflow as tf
import pytesseract
from .ocr import DetectOCR

class LicensePlateDetector:

    def __init__(self, weights, iou, score):
        # Inicializacion del objeto y asignacion de parametros
        self.input_size = 608
        self.iou = iou
        self.score = score
        # Carga el modelo guardado en el directorio custom-anchors
        self.saved_model_loaded = tf.saved_model.load(
            weights, tags=[tag_constants.SERVING])
        self.yolo_infer = self.saved_model_loaded.signatures['serving_default']
        self.processed_plates = []
        

    def processing_yolo(self, output):
        # Itera sobre los elementos clave y valor en el diccionario output
        for key, value in output.items():
            # Extrae las cajas del valor
            boxes = value[:, :, 0:4]
            # Extrae las confianzas de predicción del valor.
            pred_confidence = value[:, :, 4:]

        # Realiza la supresión de no máximos combinada en las cajas y confianzas de predicción.
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores = tf.reshape(
                pred_confidence, (tf.shape(pred_confidence)[0], -1, tf.shape(pred_confidence)[-1])),
            max_output_size_per_class = 50,
            max_total_size = 50,
            iou_threshold = self.iou,
            score_threshold = self.score
        )
        # Convierte los resultados a matrices NumPy y los devuelve.
        return [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    
    def prev_proccess(self, frame):
        # Redimension de la imagen
        data = cv2.resize(frame, (self.input_size, self.input_size))
        # Normaliza los valores de pixel en el rango [0, 1]
        data = data / 255.
        # Agrega una dimensión al array para que cumpla con el modelo
        data = data[np.newaxis, ...].astype(np.float32)
        # Convierte el tensor de datos a un tensor constante de TensorFlow
        return tf.constant(data) # dtype=tf.float32
    
    def prediction_plate(self, image: tf.Tensor):
        # Realiza una inferencia de placas utilizando el modelo cargado y la imagen de entrada.
        return self.yolo_infer(image)
    
    def draw_boxes(self, frame, boxes,day,plate_lic,scores: bool = False):

        plates_search = None
        # Itera sobre las coordenadas y puntajes de las cajas.
        for x1, y1, x2, y2, score in self.yield_coords(frame, boxes):
            value = 0
            if day == 'day':
                value = 0.90
            else:
                value = 0.50

            # solamente recortar regiones de interes con un scores mayor a 0.90
            if score > value:
                # Recorta la region de interes (ROI) de la imagen
                roi = frame[y1:y2, x1:x2]
                # Si la placa ya ha sido procesada, continúa al siguiente bucle.
                if (x1, y1, x2, y2) in self.processed_plates:
                    continue
                # Realiza la detección OCR en la región de interés.   
                detected_plate = DetectOCR().main(roi)
                
                # # ver si la placa esta en el video
                if detected_plate in plate_lic:
                    plates_search = detected_plate
                if detected_plate != None:
                    plates_search = detected_plate
                else:
                    continue

                    

                #cv2.putText(frame, plate, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2)            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}%', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2)
        return frame,plates_search
    
    def yield_coords(self, frame, boxes):
        box, scores_out, classes, num_box = boxes
        height, width, _ = frame.shape
        # Itera sobre las cajas y calcula las coordenadas relativas en la imagen original.
        for i in range(num_box[0]):
            coor = box[0][i]
            x1 = int(coor[1] * width)
            y1 = int(coor[0] * height)
            x2 = int(coor[3] * width)
            y2 = int(coor[2] * height)
            # Genera las coordenadas (x1, y1, x2, y2) y el puntaje de cada caja.
            yield x1, y1, x2, y2, scores_out[0][i]


