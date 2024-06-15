import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    cap = cv2.VideoCapture(0)  # Menggunakan kamera utama
    
    # Ambil dimensi frame dari kamera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Hitung titik nol koordinat landmark di tengah frame
    origin_x = width // 2
    origin_y = height // 2
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame.")
                break
            
            # Ubah gambar menjadi BGR agar sesuai dengan MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Proses deteksi pose
            results = pose.process(image_rgb)
            
            # Tampilkan koordinat landmark untuk beberapa bagian tubuh tertentu
            if results.pose_landmarks:
                # List indeks landmark untuk bagian-bagian tubuh yang ingin ditampilkan
                body_parts = [ 7, 8, 11, 12, 26, 25]  # Misalnya, bahu kiri, bahu kanan, dan pinggul kiri
                
                for index in body_parts:
                    # Dapatkan koordinat landmark
                    landmark = results.pose_landmarks.landmark[index]
                    landmark_x = landmark.x
                    landmark_y = landmark.y
                    
                    # Konversi koordinat ke dalam piksel
                    cx = int(landmark_x * width)
                    cy = int(landmark_y * height)
                    
                    # Hitung koordinat landmark relatif terhadap titik nol
                    relative_x = cx - origin_x
                    relative_y = cy - origin_y
                    
                    # Tampilkan koordinat landmark di sekitar landmark
                    cv2.putText(frame, f"({relative_x}, {relative_y})", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Gambar landmark pose jika ditemukan
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow('OpenPose Model', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
