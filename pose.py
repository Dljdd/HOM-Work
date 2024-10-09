from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np  

def get_keypoints(path):
    model = YOLO("yolov8n-pose.pt")  # You can choose other variants like "yolov8s-pose.pt"

    # Load the image
    image_path = "final_outputs/6/final_bg_removed_a_photo_of_WH1TE_HOM14154_900x.webp"
    image = cv2.imread(image_path)

    # Run pose estimation on the image
    results = model(image)

    # Visualize the results
    # YOLOv8 automatically draws keypoints on the image
    annotated_image = results[0].plot()
    keypoints = results[0].keypoints.cpu().numpy()
    return keypoints