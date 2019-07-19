import cv2
import numpy as np
import dlib
from math import hypot


def generateNewGlass(frame):
    landmarks = predictor(gray_frame, face)

    # Nose coordinates
    top_nose = (landmarks.part(23).x, landmarks.part(23).y)
    center_nose = (landmarks.part(27).x, landmarks.part(27).y)
    left_nose = (landmarks.part(0).x, landmarks.part(0).y)
    right_nose = (landmarks.part(16).x, landmarks.part(16).y)
    bottom_nose = (landmarks.part(30).x, landmarks.part(30).y)

    nose_width = int(hypot(left_nose[0] - right_nose[0],
                           left_nose[1] - right_nose[1])*1.1)
    nose_height = int(nose_width * 0.65)

    # New nose position
    top_left = (int(center_nose[0] - nose_width / 2),
                top_nose[1]-20)
    bottom_right = (int(center_nose[0] + nose_width / 2),
                    int(center_nose[1] + nose_height / 2))

    return nose_width, nose_height, top_left


def generateImage(frame, nose_width, nose_height, top_left):
    nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
    nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
    _, nose_mask = cv2.threshold(
        nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

    nose_area = frame[top_left[1]: top_left[1] + nose_height,
                      top_left[0]: top_left[0] + nose_width]
    nose_area_no_nose = cv2.bitwise_and(
        nose_area, nose_area, mask=nose_mask)
    final_nose = cv2.add(nose_area_no_nose, nose_pig)
    frame[top_left[1]: top_left[1] + nose_height,
          top_left[0]: top_left[0] + nose_width] = final_nose
    return frame


# Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(1)
nose_image = cv2.imread("images/glass2.png")


_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
work = True

nose_width, nose_height, top_left = None, None, None
while True:
    _, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        if(work):
            nose_width, nose_height, top_left = generateNewGlass(frame)
            frame = generateImage(frame, nose_width, nose_height, top_left)
            work = False
        else:
            frame = generateImage(frame, nose_width, nose_height, top_left)
            work = True

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
