import cv2
from deepface import DeepFace

# Load Haar Cascade untuk face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Buka kamera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break
    
    # Ubah frame ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Prediksi umur untuk setiap wajah
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Potong wajah
        try:
            result = DeepFace.analyze(face, actions=['age'], enforce_detection=False)
            age = result[0]['age']
            
            # Gambar kotak di wajah dan tampilkan umur
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print("Error:", e)

    # Tampilkan hasil
    cv2.imshow('Age Prediction', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
