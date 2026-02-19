import cv2
import mediapipe as mp
import numpy as np
import random
import time
import tkinter as tk

# Initialize MP hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
shapes = []
x_offset = 1
y_offset = 1
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()


# Spawn circle in random location
def spawn_shape(off_x, off_y, visible_w, visible_h, radius):
    x = random.randint(off_x + radius, off_x + visible_w - radius)
    y = random.randint(off_y + radius, off_y + visible_h - radius)
    timer = 0
    shapes.append((x, y, radius, timer))


def spawn_shape_victory(off_x, off_y, visible_w, visible_h, radius):
    x = random.randint(off_x + radius, off_x + visible_w - radius)
    y = random.randint(off_y + radius, off_y + visible_h - radius)
    color1 = random.randint(0, 255)
    color2 = random.randint(0, 255)
    color3 = random.randint(0, 255)
    timer = 0
    shapes.append((x, y, radius, timer, (color1, color2, color3)))


# Check if hull collides with shape
def check_collision(hull, shape):
    return cv2.pointPolygonTest(hull, (shape[0], shape[1]), False) >= 0


def resize_and_pad(frame, screen_w, screen_h):
    h, w = frame.shape[:2]
    # det scaling factor
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # resize using aspect ratio
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # black bar effect
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    # Center new image
    x_off = (screen_w - new_w) // 2
    y_off = (screen_h - new_h) // 2
    canvas[y_off: y_off + new_h, x_off: x_off + new_w] = resized

    return canvas, scale, (x_off, y_off)


def hand_track_update(cap):
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


# Alters hull coordinates to fit resized screen
def transform_hulls(hulls, scale, off_x, off_y):
    transformed_hulls = []
    for hull in hulls:
        transformed = []
        for p in hull[:, 0]:
            x = int(p[0] * scale + off_x)
            y = int(p[1] * scale + off_y)
            transformed.append([x, y])
        transformed_hulls.append(np.array(transformed, dtype=np.int32))
    return transformed_hulls


def start_screen():
    cv2.namedWindow("Bubble Pop", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Bubble Pop", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error opening webcam")

    # Initialize scaling before loop starts
    check, hulls, frame = hand_track_update(cap)
    frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)

    timer = 0

    # Compute visible webcam area
    visible_w = frame.shape[1] - 2 * off_x
    visible_h = frame.shape[0] - 2 * off_y

    # Vertical center of visible area
    center_y = int(off_y + visible_h * 0.5)
    center_y_raised = int(off_y + visible_h * 0.30)
    center_y_super_raised = int(off_y + visible_h * 0.5)

    # Horizontal thirds
    left_x = int(off_x + visible_w * (1 / 6))  # center of left third
    middle_x = int(off_x + visible_w * (3 / 6))  # center of middle third
    right_x = int(off_x + visible_w * (5 / 6))  # center of right third

    # Scaled radius
    radius = int(20 * scale)

    # Build shapes
    start_screen_shapes = [(left_x, center_y, radius, timer)]
    start_screen_shapes.append((right_x, center_y, radius, timer)),
    start_screen_shapes.append((middle_x, center_y_raised, radius, timer))

    # Label shapes
    left = start_screen_shapes[0]
    right = start_screen_shapes[1]
    middle = start_screen_shapes[2]

    # For ensuring user sees frame that option is selected. Before a user could select a mode without seeing the frame.
    # Feels bad for user.
    selected_flag = False
    easy_flag = False
    hard_flag = False
    stats_flag = False

    while True:
        if selected_flag:
            if easy_flag:
                return 1
            elif hard_flag:
                return 2
            elif stats_flag:
                return 0
            print("This shouldn't happen")
        # Initialize frame and hand locations for the frame
        check, hulls, frame = hand_track_update(cap)
        # Apply resize and bars
        frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)
        # Transform hulls to new scale
        hulls = transform_hulls(hulls, scale, off_x, off_y)

        # Handles if user does not have a hand on screen to start
        if not hulls:
            print("No hand Detected")
        else:
            # Determine if a hand is touching a circle
            for hull in hulls:
                if check_collision(hull, left):
                    selected_flag = True
                    easy_flag = True
                elif check_collision(hull, right):
                    selected_flag = True
                    hard_flag = True
                elif check_collision(hull, middle):
                    selected_flag = True
                    stats_flag = True

        # Print selection circles with corresponding labels
        # Easy
        cv2.circle(frame, (left[0], left[1]), left[2], (0, 255, 0), -1)
        (text_w, text_h), _ = cv2.getTextSize("Easy", cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, 2)
        text_x = left[0] - text_w // 2
        text_y = left[1] + left[2] + text_h + 10
        cv2.putText(frame, "Easy", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        # Hard
        cv2.circle(frame, (right[0], right[1]), right[2], (255, 0, 255), -1)
        (text_w, text_h), _ = cv2.getTextSize("Hard", cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, 2)
        text_x = right[0] - text_w // 2
        text_y = right[1] + right[2] + text_h + 10
        cv2.putText(frame, "Hard", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        # Stats
        cv2.circle(frame, (middle[0], middle[1]), middle[2], (0, 0, 255), -1)
        (text_w, text_h), _ = cv2.getTextSize("Stats", cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, 2)
        text_x = middle[0] - text_w // 2
        text_y = middle[1] + middle[2] + text_h + 10
        cv2.putText(frame, "Stats", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        # Quit Message
        (text_w, text_h), _ = cv2.getTextSize("ESC to QUIT", cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, 2)
        text_x = middle[0] - text_w // 2
        text_y = middle[1] + center_y_super_raised + text_h + 10
        cv2.putText(frame, "ESC to QUIT", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)

        cv2.imshow("Bubble Pop", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            return 3


def game_screen(selection):
    cv2.namedWindow("Bubble Pop", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Bubble Pop", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Initialize game parameters
    game_time = time.time()
    game_length = 60
    score = 0
    # Default difficulty
    timer_diff = 100
    score_diff = 35
    color = (255, 0, 0)
    # Hard difficulty
    if selection == 2:
        timer_diff = 50
        score_diff = 55
        color = (255, 0, 255)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error opening webcam")

    # Initialize scaling before loop starts
    check, hulls, frame = hand_track_update(cap)
    frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)

    # Compute visible webcam area
    visible_w = frame.shape[1] - 2 * off_x
    visible_h = frame.shape[0] - 2 * off_y

    # Scaled radius
    radius = int(15 * scale)

    while True:
        # Initialize frame and hand locations for the frame
        check, hulls, frame = hand_track_update(cap)
        # Apply resize and bars
        frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)
        # Transform hulls to new scale
        hulls = transform_hulls(hulls, scale, off_x, off_y)

        game_elapsed = time.time() - game_time
        if random.randint(1, 200) == 1 or game_elapsed - int(game_elapsed) > 0.95:
            spawn_shape(off_x, off_y, visible_w, visible_h, radius)

        for shape in shapes[:]:
            # Take info from tuple
            x_loc = shape[0]
            y_loc = shape[1]
            # Increment timer
            timer = shape[3] + 1
            # Add new timer info
            shape_new = (x_loc, y_loc, radius, timer)
            shapes.remove(shape)
            # Add new shape in
            shapes.append(shape_new)
            if timer >= timer_diff:
                # If timer expires remove the shape without increasing score
                shapes.remove(shape_new)
                continue
            if not hulls:
                print("No hand detected")
            else:
                for hull in hulls:
                    # Increment score if player hits circle
                    if check_collision(hull, shape):
                        shapes.remove(shape_new)
                        score += 1
            cv2.circle(frame, (shape[0], shape[1]), shape[2], (color[0], color[1], color[2]), -1)

        # Display stats for game
        base_x = off_x
        base_y = off_y + int(30 * scale)  # small padding from the top

        cv2.putText(frame, f"Goal: {score_diff}",
                    (base_x, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)

        cv2.putText(frame, f"Score: {score}",
                    (base_x, base_y + int(40 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)

        cv2.putText(frame, f"Time Remaining: {int(game_length - game_elapsed)}",
                    (base_x, base_y + int(80 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)

        # Win condition
        if game_elapsed >= game_length:
            if score >= score_diff:
                return True
            else:
                return False

        cv2.imshow("Bubble Pop", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            return "quit"


def end_screen(victory_flag):
    shapes.clear()
    cv2.namedWindow("Bubble Pop", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Bubble Pop", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error opening webcam")

    # Initialize scaling before loop starts
    check, hulls, frame = hand_track_update(cap)
    frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)

    # Compute visible webcam area
    visible_w = frame.shape[1] - 2 * off_x
    visible_h = frame.shape[0] - 2 * off_y

    # Scaled radius
    radius = int(15 * scale)
    while True:
        check, hulls, frame = hand_track_update(cap)
        # Apply resize and bars
        frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)
        # Spawn Shapes for victory background
        spawn_shape_victory(off_x, off_y, visible_w, visible_h, radius)
        spawn_shape_victory(off_x, off_y, visible_w, visible_h, radius)
        for shape in shapes:
            cv2.circle(frame, (shape[0], shape[1]), shape[2], (shape[4][0], shape[4][1], shape[4][2]), -1)
        if victory_flag:
            (text_w, text_h), _ = cv2.getTextSize("You win!! ESC to QUIT",
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, 2)
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            text_x = center_x - text_w // 2
            text_y = center_y + text_h // 2
            cv2.putText(frame, "You win!! ESC to QUIT", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        else:
            (text_w, text_h), _ = cv2.getTextSize("You LOSE :( ESC to QUIT",
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, 2)
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            text_x = center_x - text_w // 2
            text_y = center_y + text_h // 2
            cv2.putText(frame, "You LOSE :( ESC to QUIT", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)

        cv2.imshow('Bubble Pop', frame)

        # Quit when 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            return 0


def driver():
    selection = start_screen()
    if selection == 0:
        print("don't have these yet")
        # print_stats()
    elif selection == 3:
        print("quitting")
        return
    else:
        print("temp")
        win_flag = game_screen(selection)
        end_screen(win_flag)
    driver()


if __name__ == "__main__":
    driver()
