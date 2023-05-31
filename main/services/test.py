import cv2
from detector_matricula import LicensePlateDetector
import pytesseract


def detect_license_plate():
    # Cargar weights
    weights = f"custom-anchors/"
    iou = 0.45
    score = 0.25
    # Detector
    detection = LicensePlateDetector(weights, iou, score)
    # Cargar imagen
    cap = cv2.VideoCapture('videos/test1.mp4')

    frame_id = 0
    while True:
        return_value, frame = cap.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            break            

        # Procesado del frame
        video = detection.prev_proccess(frame)
        # Realizamos inferencia
        video_yolo = detection.prediction_plate(video)
        # Bounding boxes despues de inferencia
        boxes = detection.processing_yolo(video_yolo)
        # Mostrar predicciones
        frame_pred = detection.draw_boxes(frame, boxes, scores=True)
        cv2.imshow('frame', frame_pred)
        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    detect_license_plate()