import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MP hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start Video Feed
cap = cv2.VideoCapture(0)

# Set Frame Resolution
frame_width = 1920
frame_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


def init_screen():
    ret, frame = cap.read()
    if not ret:
        return False, None

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    # Store hand shape
    hulls = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Get the landmark coordinates
            points = np.array(
                [[int(point.x * frame.shape[1]), int(point.y * frame.shape[0])] for point in
                 hand_landmarks.landmark])

    return True, frame


def camera_loop():

    while True:
        check, frame = init_screen()

        # Render built frame
        cv2.imshow('rehabilitation', frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 0


if __name__ == "__main__":
    camera_loop()
    cap.release()
    cv2.destroyAllWindows()

