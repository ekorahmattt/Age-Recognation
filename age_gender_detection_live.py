from libraries.centroidtracker import CentroidTracker
from deepface import DeepFace
import cv2
import os
from datetime import datetime
from pytz import timezone
import pika

# Inisialisasi tracker
tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

# Folder untuk menyimpan foto
detected_faces_folder = "detected_faces"
os.makedirs(detected_faces_folder, exist_ok=True)

# Folder untuk menyimpan data
face_data_file = "face_data_log.txt"

# Inisialisasi variabel
H = None
W = None
trackableObjects = {}


connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='face_data')

    

# Kelas untuk melacak objek
class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False
        self.date, self.time = get_datetime()
        self.age = None
        self.img_path = None

# Fungsi untuk mendapatkan waktu dan tanggal
def get_datetime():
    now = datetime.now(timezone("Asia/Makassar"))
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")
    return date, time

# Fungsi untuk menyimpan data wajah
def save_face_data(to, img):
    to.date, to.time = get_datetime()
    img_path = os.path.join(detected_faces_folder, f"face_{to.objectID}_{to.date}_{to.time.replace(':', '-')}.jpg")
    cv2.imwrite(img_path, img)
    to.img_path = img_path
    massage = f"ID: {to.objectID}, Date: {to.date}, Time: {to.time}, Age: {to.age}, Image Path: {to.img_path}"
    channel.basic_publish(exchange='', routing_key='face_data', body=massage)
    # with open(face_data_file, "a") as file:
    #     file.write(f"ID: {to.objectID}, Date: {to.date}, Time: {to.time}, Age: {to.age}, Image Path: {to.img_path}\n")
    
    print(f"Saved face data - ID: {to.objectID}, Date: {to.date}, Time: {to.time}, Age: {to.age}, Image Path: {to.img_path}")

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Atur resolusi lebar
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Atur resolusi tinggi
    while True:
        try:
            success, img = cap.read()
            (H, W) = img.shape[:2]

            detecbox = []
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=4, minSize=(10, 10), maxSize=(500, 500))
            
            for (x, y, w, h) in faces:
                detecbox.append((x, y, x + w, y + h))

            obj = tracker.update(detecbox)
            
            for (objId, centroid) in obj.items():
                to = trackableObjects.get(objId, None)
                
                if to is None and len(detecbox) > 0:
                    to = TrackableObject(objId, centroid)
                    (x1, y1, x2, y2) = detecbox[min(len(detecbox)-1, list(obj.keys()).index(objId))]
                    face_img = img[y1:y2, x1:x2]
                    try:
                        analysis = DeepFace.analyze(face_img, actions=["age"], enforce_detection=False)
                        to.age = analysis[0]["age"]
                    except Exception as e:
                        to.age = "Unknown"
                    save_face_data(to, face_img)
                    
                trackableObjects[objId] = to
                
                # Tampilkan ID dan usia di frame
                age_text = f"ID {objId}, Age {to.age}"
                cv2.putText(img, age_text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.imshow("Output", img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("Error:", e)
            cap = cv2.VideoCapture(0)
            continue

    cap.release()
    cv2.destroyAllWindows()
