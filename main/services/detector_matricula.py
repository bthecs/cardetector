from tensorflow.python.saved_model import tag_constants
import numpy as np
import cv2
import tensorflow as tf
import pytesseract
from ocr import DetectOCR

class LicensePlateDetector:

    def __init__(self, weights, iou, score):
        self.input_size = 608
        self.iou = iou
        self.score = score
        self.saved_model_loaded = tf.saved_model.load(
            weights, tags=[tag_constants.SERVING])
        self.yolo_infer = self.saved_model_loaded.signatures['serving_default']
        self.processed_plates = []

    def processing_yolo(self, output):
        
        for key, value in output.items():
            boxes = value[:, :, 0:4]
            pred_confidence = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores = tf.reshape(
                pred_confidence, (tf.shape(pred_confidence)[0], -1, tf.shape(pred_confidence)[-1])),
            max_output_size_per_class = 50,
            max_total_size = 50,
            iou_threshold = self.iou,
            score_threshold = self.score
        )
        return [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    
    def prev_proccess(self, frame):
        data = cv2.resize(frame, (self.input_size, self.input_size))
        data = data / 255.
        data = data[np.newaxis, ...].astype(np.float32)
        return tf.constant(data) # dtype=tf.float32
    
    def prediction_plate(self, image: tf.Tensor):
        return self.yolo_infer(image)
    
    def draw_boxes(self, frame, boxes, scores: bool = False):
        for x1, y1, x2, y2, score in self.yield_coords(frame, boxes):
            # solamente recortar regiones de interes con un scores mayor a 0.80
            if score > 0.90:
                # Recorta la region de interes (ROI) de la imagen
                roi = frame[y1:y2, x1:x2]

                if (x1, y1, x2, y2) in self.processed_plates:
                    continue
            
                plate = DetectOCR().main(roi)
                         

                cv2.putText(frame, plate, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2)            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}%', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2)
        return frame
    
    def yield_coords(self, frame, boxes):
        box, scores_out, classes, num_box = boxes
        height, width, _ = frame.shape
        for i in range(num_box[0]):
            coor = box[0][i]
            x1 = int(coor[1] * width)
            y1 = int(coor[0] * height)
            x2 = int(coor[3] * width)
            y2 = int(coor[2] * height)
            yield x1, y1, x2, y2, scores_out[0][i]


