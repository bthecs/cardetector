import pickle
import tempfile
import cv2
from flask import Blueprint, request, jsonify, send_file
from main.services import DetectorServices
from main.services import License
import numpy as np
import threading
import concurrent.futures


detector = Blueprint('detector', __name__)

@detector.route('/', methods=['POST'])
def detect():
    video = request.files['video']
    # with open(r'main/services/videos/video.mp4', 'wb') as f:
    #     f.write(video)
    with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(video.read())
            video_path = f.name
    video_final = DetectorServices().process_video(video_path)
    return jsonify(video_final), 200 
    # return DetectorServices().process_video(video), 200

@detector.route('/matricula', methods=['POST'])
def matricula():
      video = request.files['video']
      day = request.form.get('day_night')
      plate_lic = request.form.get('plate')
      with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(video.read())
            video_path = f.name
      video_final = License().detect_license_plate(video_path, day, plate_lic)
      unique_data = []
      seen_plates = set()
      for item in video_final:
            plate = item['plate']
            if plate not in seen_plates:
                  unique_data.append(item)
                  seen_plates.add(plate)
      return jsonify(unique_data)


