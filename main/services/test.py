import json
import cv2
from .detector_matricula import LicensePlateDetector
import pytesseract
import tempfile
import time
from datetime import datetime, timedelta


class License:
    def detect_license_plate(self, video, day, plate_lic):
        start = time.time()
        # Cargar weights
        weights = f"main\services\custom-anchors"
        iou = 0.45
        score = 0.25
        # Detector
        detection = LicensePlateDetector(weights, iou, score)
        # Cargar imagen
        cap = cv2.VideoCapture(video)

        data = []
        
        frame_id = 0
        while True:
            return_value, frame = cap.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break            
            fps = cap.get(cv2.CAP_PROP_FPS)


            # Procesado del frame
            video = detection.prev_proccess(frame)
            # Realizamos inferencia
            video_yolo = detection.prediction_plate(video)
            # Bounding boxes despues de inferencia
            boxes = detection.processing_yolo(video_yolo)
            # Mostrar predicciones
            frame_pred, plate = detection.draw_boxes(frame, boxes, day, plate_lic ,scores=True)

            duration = frame_id / fps

            # Pasar duration a formato hh:mm:ss
            duration = time.strftime("%H:%M:%S", time.gmtime(duration))


            print("El vehiculo esta en el segundo {}".format(duration)+" Con la patente {}".format(plate))
            if plate is not None:
                plate_data = {
                    "plate": plate,
                    "duration": duration
                }
                data.append(plate_data)
                
            
            cv2.imshow('frame', frame_pred)
            frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        final = time.time()
        tiempo = final - start

        execution_time = timedelta(seconds=tiempo)

        execution_time_formated = str(datetime.min + execution_time)

        with open('main/logs/license_plate.txt', 'a') as f:
            f.write(f"Total de patentes detectadas {len(data)}, Tiempo de ejecucion: {execution_time_formated}\n")
        return data   


