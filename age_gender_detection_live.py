import cv2
import math
import time
import argparse
import os
import csv
import numpy as np
from datetime import datetime

# Fungsi untuk menghitung Euclidean distance
def euclidean_distance(face1, face2):
    return np.linalg.norm(face1 - face2)

def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (227, 227), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

output_dir = "captured_faces"
csv_file = "captured_data.csv"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Timestamp", "Age Category", "Image Path", "Anak-anak", "Remaja", "Dewasa", "Lansia"])

ageNet = cv2.dnn.readNet(ageModel, ageProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

cap = cv2.VideoCapture(1)
padding = 20
frame_count = 0

unique_id = 1
tracked_faces = {}  # Menyimpan data wajah yang sudah terdeteksi
capture_delay = 5  # Delay 5 detik antar capture wajah yang sama

age_counts = {"Anak-anak": 0, "Remaja": 0, "Dewasa": 0, "Lansia": 0}

def categorize_age(age_index):
    if age_index in [0, 1, 2]:
        return 'Anak-anak'
    elif age_index in [3, 4]:
        return 'Remaja'
    elif age_index in [5, 6]:
        return 'Dewasa'
    else:
        return 'Lansia'

while True:
    t = time.time()
    hasFrame, frame = cap.read()

    if not hasFrame:
        cv2.waitKey(1)
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frameFace, bboxes = getFaceBox(faceNet, small_frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    current_time = time.time()

    for bbox in bboxes:
        face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        
        if face.size == 0:
            continue
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        same_face_detected = False
        for face_id, (prev_blob, last_time) in tracked_faces.items():
            if euclidean_distance(prev_blob.flatten(), blob.flatten()) < 50 and (current_time - last_time) < capture_delay:
                same_face_detected = True
                break

        if same_face_detected:
            print("Same face detected recently, skipping capture.")
            continue
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age_index = agePreds[0].argmax()
        age_category = categorize_age(age_index)
        age_counts[age_category] += 1
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        face_filename = os.path.join(output_dir, f"face_{unique_id}.jpg")
        cv2.imwrite(face_filename, face)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([unique_id, timestamp, age_category, face_filename, age_counts['Anak-anak'], age_counts['Remaja'], age_counts['Dewasa'], age_counts['Lansia']])

        print("Saved data: ID={}, Timestamp={}, Age Category={}, Image Path={}, Counts={}".format(unique_id, timestamp, age_category, face_filename, age_counts))
        
        label = "{}".format(age_category)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Age Detection Demo", frameFace)

        tracked_faces[unique_id] = (blob, current_time)
        unique_id += 1

    print("time : {:.3f}".format(time.time() - t))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
