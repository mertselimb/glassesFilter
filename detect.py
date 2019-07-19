import cv2
import numpy as np
import dlib
from math import hypot,degrees,atan2


def generateNewGlass(frame):
    landmarks = predictor(gray_frame, face)

    top_glass = (landmarks.part(23).x, landmarks.part(23).y)
    center_glass = (landmarks.part(27).x, landmarks.part(27).y)
    left_glass = (landmarks.part(0).x, landmarks.part(0).y)
    right_glass = (landmarks.part(16).x, landmarks.part(16).y)
    bottom_glass = (landmarks.part(30).x, landmarks.part(30).y)

    radian = atan2(bottom_glass[1]-center_glass[1],bottom_glass[0]-center_glass[0])
    degree = degrees(radian)-90

    glass_width = int(hypot(left_glass[0] - right_glass[0],
                           left_glass[1] - right_glass[1])*1.1)
    glass_height = int(glass_width * 0.65)

    top_left = (int(center_glass[0] - glass_width / 2),
                top_glass[1]-20)
    bottom_right = (int(center_glass[0] + glass_width / 2),
                    int(center_glass[1] + glass_height / 2))

    return glass_width, glass_height, top_left, degree


def generateImage(frame, glass_width, glass_height, top_left, degree):
    resized_glass = cv2.resize(glass_image, (glass_width, glass_height))
    resized_glass_gray = cv2.cvtColor(resized_glass, cv2.COLOR_BGR2GRAY)
    _, glass_mask = cv2.threshold(
        resized_glass_gray, 25, 255, cv2.THRESH_BINARY_INV)

    glass_area = frame[top_left[1]: top_left[1] + glass_height,
                      top_left[0]: top_left[0] + glass_width]
    glass_area_no_glass = cv2.bitwise_and(
        glass_area, glass_area, mask=glass_mask)
    final_glass = cv2.add(glass_area_no_glass, resized_glass)
    frame[top_left[1]: top_left[1] + glass_height,
          top_left[0]: top_left[0] + glass_width] = final_glass
    return frame


cap = cv2.VideoCapture(0)
glass_image = cv2.imread("images/glass2.png")


_, frame = cap.read()
rows, cols, _ = frame.shape
glass_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    glass_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        glass_width, glass_height, top_left, degree = generateNewGlass(frame)
        frame = generateImage(frame, glass_width, glass_height, top_left, degree)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
