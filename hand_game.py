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


# Initialize screen and hand tracking functions
def init_screen():
    ret, frame = cap.read()
    if not ret:
        return False, None, None

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

    return True, hulls, frame


def start_screen():
    # Menu shape dimensions
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
        # Initialize frame and hand tracking
        check, hulls, frame = init_screen()

        # Label specific shapes for proper result
        left = start_screen_shapes[0]
        right = start_screen_shapes[1]
        close = start_screen_shapes[2]

        for hull in hulls:
            # Check collisions for menu selection
            if check_collision(hull, left):
                return 1
            elif check_collision(hull, right):
                return 2
            elif check_collision(hull, close):
                return 0
        else:
            # Draw the circles and make the titles
            cv2.circle(frame, (left[0], left[1]), left[2], (0, 255, 0), -1)
            cv2.putText(frame, f"Easy", (left[0] - int(0.5 * left[2]), left[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.circle(frame, (right[0], right[1]), right[2], (255, 0, 255), -1)
            cv2.putText(frame, f"Hard", (right[0] - int(0.5 * right[2]), right[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.circle(frame, (close[0], close[1]), close[2], (0, 0, 255), -1)
            cv2.putText(frame, f"Exit", (close[0] - int(0.5 * right[2]), close[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Render built frame
        cv2.imshow('Balloon Popper', frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 0


def end_screen(victory_flag):
    while True:
        check, hulls, frame = init_screen()
        spawn_shape_victory()
        for shape in shapes:
            cv2.circle(frame, (shape[0], shape[1]), shape[2], (255, 0, 0), -1)
        if victory_flag:
            cv2.putText(frame, f"YOU WIN!!!! Press 'q' to QUIT", (int(frame_width / 5), int(frame_height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        else:
            cv2.putText(frame, f"You Lose :( Press 'q' to QUIT", (int(frame_width / 5), int(frame_height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        cv2.imshow('Balloon Popper', frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 0


def run(selected_difficulty):
    game_time = time.time()
    game_length = 60
    score = 0
    timer_diff = None
    score_diff = None
    color = None
    if selected_difficulty == 1:
        timer_diff = 100
        score_diff = 35
        color = (255, 0, 0)
    elif selected_difficulty == 2:
        timer_diff = 50
        score_diff = 55
        color = (255, 0, 255)
    elif selected_difficulty == 0:
        return
    if timer_diff is None:
        return
    # Game loop
    while True:
        check, hulls, frame = init_screen()
        game_elapsed = time.time() - game_time
        # Spawn in shapes, random rate, plus one is set to spawn every second or so, at least
        if random.randint(1, 50) == 1 or game_elapsed - int(game_elapsed) > 0.95:
            spawn_shape()
        # Check shapes for collisions to remove them
        for shape in shapes[:]:
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
        cv2.putText(frame, f"Goal: {score_diff}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"Time Remaining: {int(game_length - game_elapsed)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Win condition
        if game_elapsed >= game_length:
            if score >= score_diff:
                return True
            else:
                return False

        # Display the resulting frame
        cv2.imshow('Balloon Popper', frame)

        # Press q to exit at any time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    difficulty = start_screen()
    result = run(difficulty)
    end_screen(result)
    cap.release()
    cv2.destroyAllWindows()
