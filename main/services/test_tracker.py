import uuid
import cv2
import numpy as np
from collections import OrderedDict

from object_detection import ObjectDetection


class VehicleTracker:
    def __init__(self, disappearance_time=10):
        self.disappearance_time = disappearance_time
        self.active_vehicles = OrderedDict()

    def is_overlap(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        if x3 > x2 or x4 < x1 or y3 > y2 or y4 < y1:
            return False
        return True

    def update_vehicles(self, frame):
        for vehicle_id in list(self.active_vehicles.keys()):
            vehicle = self.active_vehicles[vehicle_id]
            success, box = vehicle['tracker'].update(frame)
            if success:
                box = tuple(map(int, box))
                vehicle['box'] = box
                vehicle['disappearance_frames'] = 0
            else:
                vehicle['disappearance_frames'] += 1
                if vehicle['disappearance_frames'] >= self.disappearance_time:
                    if not vehicle['counted']:
                        vehicle['counted'] = True
                        self.active_vehicles.move_to_end(vehicle_id, last=False)  # Move to beginning
                    else:
                        del self.active_vehicles[vehicle_id]
                else:
                    self.active_vehicles.move_to_end(vehicle_id, last=True)  # Move to end

    def process_video(self, video_path):
        detector = ObjectDetection()
        cap = cv2.VideoCapture(video_path)
        self.active_vehicles = OrderedDict()
        frames = 0
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            (class_ids, scores, boxes) = detector.detect(frame)
            for box in boxes:
                x,y,w,h = box
                box_new = (x, y, x + w, y + h)
                vehicle_detected = False
                for vehicle_id in self.active_vehicles:
                    if self.is_overlap(box_new, self.active_vehicles[vehicle_id]['box']):
                        vehicle_detected = True
                        self.active_vehicles[vehicle_id]['disappearance_frames'] = 0
                        break
                if not vehicle_detected:
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, box_new)
                    vehicle_id = str(uuid.uuid4())
                    self.active_vehicles[vehicle_id] = {
                        'box': box_new,
                        'tracker': tracker,
                        'disappearance_frames': 0,
                        'counted': False
                    }
            self.update_vehicles(frame)
            for vehicle in self.active_vehicles.values():
                x, y, w, h = vehicle['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count = sum([1 for v in self.active_vehicles.values() if not v['counted']])
            cv2.putText(frame, f'Vehículos activos: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

                

if __name__ == '__main__':
    tracker = VehicleTracker()
    video_path = 'videos/carros-1900.mp4'
    frames, count = tracker.process_video(video_path)
    print(f'Cantidad de frames: {frames}')
    print(f'Cantidad de vehículos: {count}')