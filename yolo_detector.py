import threading
import cv2
import numpy as np
import os
import time
import requests # For downloading model
from tqdm import tqdm # For download progress bar

from ultralytics import YOLO # Import YOLO directly from ultralytics
from qvl.qcar import QLabsQCar # Keep QLabsQCar as per main.py's usage

class YOLODetector(threading.Thread):
   