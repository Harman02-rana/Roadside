from ultralytics import YOLO
import cv2

# load model
model = YOLO("yolov8n.pt")

# run detection
results = model("test.jpeg")

for r in results:
    img = r.plot()

    # resize image to bigger size
    img = cv2.resize(img, (1000, 600))

    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Detection completed")