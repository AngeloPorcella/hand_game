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


def start_screen():
    # Menu shapes
    start_screen_shapes = []
    left_width = int(frame_width * 0.1)
    right_width = int(frame_width * 0.78)
    height = int(frame_height * 0.5)
    radius = int(frame_width / 30)
    timer = 0
    start_screen_shapes.append((left_width, height, radius, timer))
    start_screen_shapes.append((right_width, height, radius, timer))
    start_screen_shapes.append((int(frame_width * 0.45), int(frame_height * 0.75), radius, timer))
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
                    [[int(point.x * frame.shape[1]), int(point.y * frame.shape[0])] for point in
                     hand_landmarks.landmark])
                # Draw the convex hull
                hull = cv2.convexHull(points)
                cv2.fillPoly(frame, [hull], color=(0, 0, 255))
                hulls.append(hull)
        left = start_screen_shapes[0]
        right = start_screen_shapes[1]
        close = start_screen_shapes[2]

        for hull in hulls:
            # Increment score if player hits circle
            if check_collision(hull, left):
                return 1
            elif check_collision(hull, right):
                return 2
            elif check_collision(hull, close):
                return 0
        else:
            # Draw the circles and make the titles
            cv2.circle(frame, (left[0], left[1]), left[2], (0, 255, 0), -1)
            cv2.putText(frame, f"Easy", (left[0] - int(0.5 * left[2]), left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.circle(frame, (right[0], right[1]), right[2], (255, 0, 255), -1)
            cv2.putText(frame, f"Hard", (right[0] - int(0.5 * right[2]), right[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.circle(frame, (close[0], close[1]), close[2], (0, 0, 255), -1)
            cv2.putText(frame, f"Exit", (close[0] - int(0.5 * right[2]), close[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # cv2.circle(frame, (left[0], left[1]), left[2], (255, 0, 255), -1)
        # cv2.circle(frame, (right[0], right[1]), right[2], (255, 0, 255), -1)

        cv2.imshow('Balloon Popper', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 0


def run(selected_difficulty):
    win_flag = False
    start_time = None
    score = 0
    timer_diff = None
    score_diff = None
    color = None
    if selected_difficulty == 1:
        timer_diff = 100
        score_diff = 15
        color = (255, 0, 0)
    elif selected_difficulty == 2:
        timer_diff = 50
        score_diff = 25
        color = (255, 0, 255)
    elif selected_difficulty == 0:
        return
    if timer_diff is None:
        return
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
                cv2.circle(frame, (shape[0], shape[1]), shape[2], (color[0], color[1], color[2]), -1)
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
                if timer >= timer_diff:
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
                    cv2.circle(frame, (shape[0], shape[1]), shape[2], (color[0], color[1], color[2]), -1)
        # Display Score
        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Win condition
        if score >= score_diff:
            if not win_flag:
                start_time = time.time()
            win_flag = True
            cv2.putText(frame, f"YOU WIN!!!!", (int(frame_width / 3), int(frame_height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            elapsed = time.time() - start_time
            if elapsed >= 5:
                break
        # Display the resulting frame
        cv2.imshow('Balloon Popper', frame)

        # Press q to exit at any time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    difficulty = start_screen()
    run(difficulty)
