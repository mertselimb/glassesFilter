import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
glass_image = cv2.imread("glass1.png")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    image_resized = glass_image
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(gray, face)

        top_area = (landmarks.part(43).x, landmarks.part(43).y)
        center_area = (landmarks.part(27).x, landmarks.part(27).y)
        left_area = (landmarks.part(0).x, landmarks.part(0).y)
        right_area = (landmarks.part(16).x, landmarks.part(16).y)
        bottom_area = (landmarks.part(30).x, landmarks.part(30).y)

        area_width = int(
            hypot(left_area[0] - right_area[0], left_area[1] - right_area[1]))
        area_height = int(
            hypot(top_area[0] - bottom_area[0], top_area[1] - bottom_area[1]))

        top_left = (int(center_area[0] - area_width / 2),
                    int(center_area[1] - area_height / 2))
        bottom_right = (int(center_area[0] + area_width / 2),
                        int(center_area[1] + area_height / 2))

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        image_resized = cv2.resize(glass_image, (area_width, area_height))

        image_resized_gray = image_resized
        _, glass_image = cv2.threshold(
            image_resized_gray, 25, 255, cv2.THRESH_BINARY_INV)

        glass_area = frame[top_left[1]: top_left[1] + area_height,
                           top_left[0]: top_left[0] + area_width]

        glass_area_inv = cv2.bitwise_and(
            glass_area, glass_area, mask=glass_image)

        final_image = cv2.add(glass_area_inv, image_resized)

        frame[top_left[1]: top_left[1] + area_height,
              top_left[0]: top_left[0] + area_width] = final_image

    cv2.imshow("Frame", frame)
    # cv2.imshow("Resized", image_resized)

    key = cv2.waitKey(1)
    if key == 27:
        break
