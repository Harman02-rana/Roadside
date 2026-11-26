import cv2
import numpy as np

def check_illumination(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    print("Brightness:", brightness)

    if brightness < 50:
        return "Low Light (High Risk)"
    elif brightness < 120:
        return "Medium Light (Medium Risk)"
    else:
        return "Good Light (Low Risk)"


# test
result = check_illumination("test_images/img-399.jpg")
print("Illumination:", result)