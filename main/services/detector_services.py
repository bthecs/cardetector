import cv2
import numpy as np
from object_detection import ObjectDetection


class DetectorServices:

    def __init__(self):
        self.car_count = 0
        self.centroids = []
        self.tracker_list = []

    def count_vehicles(self):
        # Count the number of unique vehicles
        unique_centroids = []
        for c in self.centroids:
            is_unique = True
            for uc in unique_centroids:
                dist = np.sqrt((c[0] - uc[0]) ** 2 + (c[1] - uc[1]) ** 2)
                if dist < 50:  # If a centroid is closer than 50 pixels to another centroid, consider it the same vehicle
                    is_unique = False
                    break
            if is_unique:
                unique_centroids.append(c)

        return len(unique_centroids)

    def process_video(self):
        # Initialize the classifier
        od_model = ObjectDetection()

        # Get the video
        video = cv2.VideoCapture(r"./videos/video.mp4")

        # Get the frame rate of the video
        fps = video.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = video.read()

            if ret:
                # Detect objects on frame
                (class_ids, scores, boxes) = od_model.detect(frame)

                for box in range(len(boxes)):
                    if class_ids[box] == 2:
                        # Get the centroid of the vehicle
                        (x, y, w, h) = boxes[box]
                        centroid = (x + w//2, y + h//2)

                        # Check if the centroid is close to another centroid already in the list
                        is_unique = True
                        for c in self.centroids:
                            dist = np.sqrt((centroid[0] - c[0]) ** 2 + (centroid[1] - c[1]) ** 2)
                            if dist < 50:
                                is_unique = False
                                break

                        if is_unique:
                            # Increment the number of unique cars found
                            self.car_count += 1

                            # Add the centroid to the list
                            self.centroids.append(centroid)

                           
                        # Draw bounding box and centroid
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.circle(frame, centroid, 4, (0, 255, 0 ), -1)

                # Count vehicles
                vehicle_count = self.count_vehicles()
                print(f"Number of unique vehicles: {vehicle_count}")

                # Draw the vehicle count on the frame
                cv2.putText(frame, f"Vehicle count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Show the frame
                cv2.imshow("Vehicle Detection", frame)
                cv2.waitKey(1)

            else:
                break

        # Release the video capture object and close all windows
        video.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    detector = DetectorServices()
    detector.process_video()


# tracker_name = str(tracker).split()[0][1:]
# cap = cv2.VideoCapture(0)
# ret,frame = cap.read()
# roi = cv2.selectROI(frame,False)

# ret = tracker.init(frame,roi)

# while True:
#     ret, frame = cap.read()
#     success,roi = tracker.update(frame)
#     (x,y,w,h) = tuple(map(int,roi))
#     if success:
#         p1 = (x,y)
#         p2 = (x+w,y+h)
#         cv2.rectangle(frame,pt1=p1,pt2=p2,color=(0,255,0),thickness=3)
#         roi = frame[y:y+h,x:x+w]
#     else:
#         cv2.putText(frame,text="Failure to Detect Track