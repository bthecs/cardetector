import tempfile
import cv2
from flask import Blueprint, request, jsonify
from main.services import DetectorServices
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
    thread = threading.Thread(target=DetectorServices().process_video, args=(video_path,))
    thread.start()

    return "Video received", 200 
    # return DetectorServices().process_video(video), 200
    
