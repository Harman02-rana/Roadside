from ultralytics import YOLO
import cv2

# Load models
pothole_model = YOLO("runs/detect/train2/weights/best.pt")
garbage_model = YOLO("runs/detect/train3/weights/best.pt")

# Image path
image_path = "test_images/img-399.jpg"

# Read image
img = cv2.imread(image_path)

# Pothole detection
pothole_results = pothole_model(image_path)

for box in pothole_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, "Pothole", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# Garbage detection
garbage_results = garbage_model(image_path)

for box in garbage_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, "Garbage", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# Show image
cv2.imshow("Combined Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()