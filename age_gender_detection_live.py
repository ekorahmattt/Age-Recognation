import cv2
import math
import time
import argparse
import os

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

# Create directory to save captured images
output_dir = "captured_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)


cap = cv2.VideoCapture(1)
padding = 20
frame_count = 0

# Define age group labels
def categorize_age(age_index):
    """ Categorize age into general groups. """
    # Categorize based on ageList indices
    if age_index in [0, 1, 2]:  # (0-2), (4-6), (8-12)
        return 'Anak-anak'
    elif age_index in [3, 4]:  # (15-20), (25-32)
        return 'Remaja'
    elif age_index in [5, 6]:  # (38-43), (48-53)
        return 'Dewasa'
    else:  # (60-100)
        return 'Lansia'

while True:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()

    if not hasFrame:
        cv2.waitKey(1)
        break

    # Creating a smaller frame for better optimization
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frameFace, bboxes = getFaceBox(faceNet, small_frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        
        # Check if the face image is valid
        if face.size == 0:
            continue
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age_index = agePreds[0].argmax()  # Get the index of the maximum age prediction
        age_category = categorize_age(age_index)  # Categorize the age into one of the general groups
        
        print("Age Output : {}".format(agePreds))
        print("Age Category : {}, conf = {:.3f}".format(age_category, agePreds[0].max()))

        label = "{}".format(age_category)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.namedWindow("Age Detection Demo", cv2.WINDOW_NORMAL)  # Membuat window yang bisa diubah ukurannya
        cv2.resizeWindow("Age Detection Demo", 800, 600)  # Mengatur ukuran awal window
        cv2.imshow("Age Detection Demo", frameFace)

        # # Save the captured face
        # face_filename = os.path.join(output_dir, f"face_{frame_count}.jpg")
        # cv2.imwrite(face_filename, face)
        # frame_count += 1

    print("time : {:.3f}".format(time.time() - t))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
