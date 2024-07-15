import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def light_reflection_analysis(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            return True
    return False


def texture_analysis(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()

    print(f"对比度: {contrast}, 相异性: {dissimilarity}, 同质性: {homogeneity}, 能量: {energy}")

    # 调整阈值
    if 60 < contrast < 150 and 4 < dissimilarity < 5 and homogeneity > 0.3 and energy > 0.03:
        return True
    else:
        return False


def combined_analysis(frame, roi):
    reflection = light_reflection_analysis(roi)
    texture = texture_analysis(roi)
    return reflection and texture


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]

        if combined_analysis(frame, roi):
            label = "Alive"
            color = (0, 255, 0)
        else:
            label = "Dead"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
