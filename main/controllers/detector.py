import cv2
from flask import Blueprint, request, jsonify
from main.services import DetectorServices
from io import BytesIO
import numpy as np

detector = Blueprint('detector', __name__)

@detector.route('/', methods=['POST'])
def detect():
    video = request.files['video'].read()
    with open(r'main/services/videos/video.mp4', 'wb') as f:
        f.write(video)
    
    return DetectorServices().process_video(video), 200
    
