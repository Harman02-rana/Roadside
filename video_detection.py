from ultralytics import YOLO
import cv2

# Load models
pothole_model = YOLO("runs/detect/train2/weights/best.pt")
garbage_model = YOLO("runs/detect/train3/weights/best.pt")

# Open video
cap = cv2.VideoCapture("test_videos/road.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pothole detection
    pothole_results = pothole_model(frame)

    for box in pothole_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Pothole", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Garbage detection
    garbage_results = garbage_model(frame)

    for box in garbage_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Garbage", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show frame
    cv2.imshow("Video Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()