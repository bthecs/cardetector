import io
import tempfile
import cv2
import numpy as np
import torch
import time
from datetime import datetime, timedelta


class DetectorServices:

    def __init__(self):
        self.vehicle_count = 0
        self.tracked_vehicles = {}

    def distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def process_video(self, video):
        start = time.time()
        # Cargar el modelo de YOLOv5 pre-entrenado
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Creamos el objeto para leer el video
        cap = cv2.VideoCapture(video)

        # Iteramos sobre todos los cuadros del video
        while cap.isOpened():
            # Iniciar un contador de tiempo

            # Leemos el cuadro actual
            ret, frame = cap.read()
            if not ret:
                break

            # Detectamos los objetos en el cuadro utilizando YOLOv5
            results = model(frame)

            # Recorremos los resultados de detección y rastreamos los vehículos
            for obj in results.xyxy[0]:
                if obj[-1] == 2:  # Verificamos si el objeto es un vehículo
                    x1, y1, x2, y2 = map(int, obj[:4])
                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculamos el centroide del vehículo

                    # Buscamos el vehículo más cercano al centroide actual
                    min_distance = float('inf')
                    nearest_vehicle_id = None
                    for vehicle_id, last_centroid in self.tracked_vehicles.items():
                        dist = self.distance(centroid, last_centroid)
                        if dist < min_distance:
                            min_distance = dist
                            nearest_vehicle_id = vehicle_id

                    # Si el vehículo más cercano está lo suficientemente cerca, lo asociamos al vehículo actual
                    if nearest_vehicle_id is not None and min_distance < 60:
                        self.tracked_vehicles[nearest_vehicle_id] = centroid
                    # Si no hay vehículos cercanos, lo consideramos un vehículo nuevo y lo agregamos al diccionario
                    else:
                        self.tracked_vehicles[len(self.tracked_vehicles)] = centroid
                        self.vehicle_count += 1

            # Mostramos el cuadro actual con los centroides y la cantidad de vehículos detectados
            for vehicle_id, centroid in self.tracked_vehicles.items():
                cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
                cv2.putText(frame, f'ID: {vehicle_id}', (centroid[0] + 10, centroid[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f'VEHICLES: {self.vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video', frame)
            # Esperamos la tecla 'q' para salir del bucle
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        # Liberamos la captura y cerramos todas las ventanas
        cap.release()
        cv2.destroyAllWindows()
        # finalizar el contador de tiempo
        final = time.time()

        tiempo = final - start

        execution_time = timedelta(seconds=tiempo)

        execution_time_formated = str(datetime.min + execution_time)

        data = [{
            "Total de vehiculos en el video": self.vehicle_count
        }]

        # Crear log de sistema
        with open('main/logs/census.txt', 'a') as f:
            f.write(f"Total de vehiculos en el video: {self.vehicle_count}, Tiempo que tardo el sistema: {execution_time_formated[11:]} \n")
        
        return data
        

        
