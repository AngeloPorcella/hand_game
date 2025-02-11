import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MP hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

shapes = []


# Start Video Feed
cap = cv2.VideoCapture(0)

# Set Frame Resolution
frame_width = 1920
frame_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Spawn circle in random location
def spawn_shape():
    x = random.randint(50, frame_width - 50)
    y = random.randint(50, frame_height - 50)
    radius = 20
    timer = 0
    shapes.append((x, y, radius, timer))


def spawn_shape_victory():
    x = random.randint(0, frame_width)
    y = random.randint(0, frame_height)
    radius = 20
    timer = 0
    shapes.append((x, y, radius, timer))


# Check if hull collides with shape
def check_collision(hull, shape):
    return cv2.pointPolygonTest(hull, (shape[0], shape[1]), False) >= 0


def run():
    win_flag = False
    start_time = None
    score = 0
    # Game loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        # Store hand shape
        hulls = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the landmark coordinates
                points = np.array(
                    [[int(point.x * frame.shape[1]), int(point.y * frame.shape[0])] for point in hand_landmarks.landmark])
                # Draw the convex hull
                hull = cv2.convexHull(points)
                cv2.fillPoly(frame, [hull], color=(0, 0, 255))
                hulls.append(hull)

        if random.randint(1, 50) == 1:
            spawn_shape()
        for shape in shapes[:]:
            if win_flag:
                spawn_shape_victory()
                cv2.circle(frame, (shape[0], shape[1]), shape[2], (255, 0, 255), -1)
            else:
                # Take info from tuple
                x_loc = shape[0]
                y_loc = shape[1]
                rad = shape[2]
                # Increment timer
                timer = shape[3] + 1
                # Add new timer info
                shape_new = (x_loc, y_loc, rad, timer)
                shapes.remove(shape)
                # Add new shape in
                shapes.append(shape_new)
                if timer >= 75:
                    # If timer expires remove the shape without increasing score
                    shapes.remove(shape_new)
                    continue
                for hull in hulls:
                    # Increment score if player hits circle
                    if check_collision(hull, shape):
                        shapes.remove(shape_new)
                        score += 1
                        break
                else:
                    cv2.circle(frame, (shape[0], shape[1]), shape[2], (255, 0, 255), -1)
        # Display Score
        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Win condition
        if score >= 10:
            if not win_flag:
                start_time = time.time()
            win_flag = True
            cv2.putText(frame, f"YOU WIN!!!!", (int(frame_width / 3), int(frame_height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            elapsed = time.time() - start_time
            if elapsed >= 5:
                break
        # Display the resulting frame
        cv2.imshow('Hand Detection', frame)

        # Press q to exit at any time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    run()
