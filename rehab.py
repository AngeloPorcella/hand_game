import matplotlib
matplotlib.use("Agg")
import cv2
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import numpy as np
import tkinter as tk

# Setup gesture recognizer task components
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

# List to track recently recognized gestures (15 frames)
results = []
# Last 2 gestures chosen
pulled_gestures = []

celebration_list = ["Yay!",
                    "Good Job!",
                    "Keep Going!",
                    "Booya!",
                    "Sick!",
                    "Right On!",
                    "You Rock!",
                    "Huzzah!",
                    "Excelsior!",
                    "Hurray!",
                    "Yippie!!!",
                    "You got this!",
                    "YeeHaw!",
                    "WooHoo!"
                    ]

# Hard difficulty, involves more complicated hand movements
# and requires mobilization of the arm/wrist
gesture_dict_hard = {"Closed_Fist": "Closed Fist",
                     "Open_Palm": "High Five",
                     "Pointing_Up": "Point Up",
                     "Thumb_Down": "Thumbs Down",
                     "Thumb_Up": "Thumbs Up",
                     "Victory": "Peace Sign",
                     "ILoveYou": "I Love You"}

# Medium difficulty, more complex hand motions that can be performed with
# little arm/wrist mobilization
gesture_dict_medium = {"Closed_Fist": "Closed Fist",
                       "Open_Palm": "High Five",
                       "Pointing_Up": "Point Up",
                       "Thumb_Up": "Thumbs Up",
                       "Victory": "Peace Sign"}

# Easy difficulty, gripping motion
gesture_dict_easy = {"Closed_Fist": "Closed Fist",
                     "Open_Palm": "High Five"}


def get_gesture_dict(difficulty):
    if difficulty == "easy":
        return gesture_dict_easy
    elif difficulty == "med":
        return gesture_dict_medium
    elif difficulty == "hard":
        return gesture_dict_hard


def get_gesture_list(difficulty):
    if difficulty == "easy":
        return list(gesture_dict_easy.keys())
    elif difficulty == "med":
        return list(gesture_dict_medium.keys())
    elif difficulty == "hard":
        return list(gesture_dict_hard.keys())


def get_gesture_list_names(difficulty):
    if difficulty == "easy":
        return list(gesture_dict_easy.values())
    elif difficulty == "med":
        return list(gesture_dict_medium.values())
    elif difficulty == "hard":
        return list(gesture_dict_hard.values())


def append_to_tbg_avg_csv(data_list, difficulty):
    avg_session_tbg = average_list(data_list)
    if avg_session_tbg is 0:
        print("No data recorded")
        return 0
    with open('avg_tbg_' + difficulty + '.csv', 'a') as file:
        file.write(str(avg_session_tbg) + "\n")  # newline instead of comma+space


def append_to_score_csv(score, difficulty):
    with open('score_' + difficulty + '.csv', 'a') as file:
        file.write(str(score) + "\n")  # newline instead of comma+space


def append_to_total(data_point):
    print("This is where I will append to a file for future graphing")


# Average datapoints
def average_list(data_list):
    if len(data_list) < 1:
        return 0
    total = 0
    for item in data_list:
        total += float(item)
    return total / len(data_list)


# ensures list has context window of 15 frames for some future uses I cannot forsee
def add_to_list(item):
    size = len(results)
    if size >= 15:
        results.pop(0)
    results.append(item)


# length 2 buffer to ensure no duplicate pulls
def add_to_pulled_gestures(item):
    size = len(pulled_gestures)
    if size > 3:
        pulled_gestures.pop(0)
    pulled_gestures.append(item)


# get the most recently recognized gesture
def get_latest_result():
    if len(results) == 0:
        return "empty"
    return results[-1]


def clear_results():
    results.clear()


def get_latest_pulled_gesture():
    if len(pulled_gestures) == 0:
        return "empty"
    return pulled_gestures[-1]


def pick_next_gesture(gesture_list, easy_flag):
    # If the chosen gesture has been performed recently, keep trying until a new choice is chosen
    # Handling easy mode with 2 gestures
    if easy_flag:
        if get_latest_pulled_gesture() == "Open_Palm":
            chosen_gesture = "Closed_Fist"
        else:
            chosen_gesture = "Open_Palm"
        add_to_pulled_gestures(chosen_gesture)
    else:
        chosen_gesture = random.choice(gesture_list)
        if get_latest_pulled_gesture() == "empty":
            add_to_pulled_gestures("Thumb_up")
            add_to_pulled_gestures("Pointing_Up")
        if chosen_gesture in pulled_gestures:
            while chosen_gesture in pulled_gestures:
                chosen_gesture = random.choice(gesture_list)
        add_to_pulled_gestures(chosen_gesture)
    return chosen_gesture, time.time()


def check_gesture():
    pulled_gesture = get_latest_pulled_gesture()
    result_gesture = get_latest_result()
    if pulled_gesture == result_gesture:
        return True, time.time()
    else:
        return False, None


def time_diff(most_recent_time, earlier_time):
    return most_recent_time - earlier_time


# print results list
def print_result_full():
    for result in results:
        print("in list:" + result)


# Callback to receive results
def print_result(result: vision.GestureRecognizerResult, unused_image, timestamp):
    if result.gestures:
        top_gesture = result.gestures[0][0]
        print(f"Gesture: {top_gesture.category_name} ({top_gesture.score:.2f})")
        return top_gesture.category_name


# What is called in method to perform gesture checks
def handle_result(result: vision.GestureRecognizerResult, unused_image, timestamp):
    if result.gestures:
        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name
        add_to_list(gesture_name)


def create_graph_from_csvs(score):

    files = {}
    max_len = 0

    screen_width, screen_height = get_monitor_height_width()

    dpi = 100

    if score:
        files = {
            "Easy": "score_easy.csv",
            "Medium": "score_med.csv",
            "Hard": "score_hard.csv"
        }
    else:
        files = {
            "Easy": "avg_tbg_easy.csv",
            "Medium": "avg_tbg_med.csv",
            "Hard": "avg_tbg_hard.csv"
        }

    data_found = False

    fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.set_title("Average Time Between Gestures (ESC to QUIT)")
    ax.set_xlabel("Session")
    ax.set_ylabel("Avg Time")

    colors = {
        "Easy": "green",
        "Medium": "blue",
        "Hard": "red"
    }

    for label, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path, header=None, names=['avg_tbg'])

            if not df.empty:
                y_values = df['avg_tbg'].astype(float).values
                x_values = np.arange(1, len(y_values) + 1)

                ax.plot(x_values, y_values, marker='o', color=colors[label], label=label)
                max_len = max(max_len, len(y_values))
                data_found = True

    if not data_found:
        print("No data found in any CSV. Play a game first.")
        return None

    ax.set_xticks(np.arange(1, max_len + 1, 1))
    ax.set_xlim(1, max_len)

    ax.legend()
    plt.tight_layout()

    # Convert figure to OpenCV image
    canvas = FigureCanvas(fig)
    canvas.draw()

    width, height = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = buf.reshape((height, width, 4))

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image_bgr


def get_monitor_height_width():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


def display_graph_screen():
    cv2.destroyAllWindows()
    cv2.namedWindow("Stats Graph", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Stats Graph", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_width, screen_height = get_monitor_height_width()
    graph_img = create_graph_from_csvs(score=True)
    graph_img = cv2.resize(graph_img, (screen_width, screen_height))
    if graph_img is not None:
        cv2.imshow("Stats Graph", graph_img)
        cv2.waitKey(0)
        return
    return


def delete_csv(score):
    files = []
    if score:
        files = ["score_easy.csv", "score_med.csv", "score_hard.csv"]
    else:
        files = ["avg_tbg_easy.csv", "avg_tbg_med.csv", "avg_tbg_hard.csv"]
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} deleted (stats reset).")
        else:
            print(f"{file_path} not found (nothing to reset).")


def resize_and_pad(frame, screen_width, screen_height):
    h, w = frame.shape[:2]
    # det scaling factor
    scale = min(screen_width / w, screen_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # resize using aspect ratio
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # black bar effect
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    # Center new image
    x_offset = (screen_width - new_w) // 2
    y_offset = (screen_height - new_h) // 2
    canvas[y_offset: y_offset + new_h, x_offset: x_offset + new_w] = resized

    return canvas, scale, (x_offset, y_offset)


def start_screen():

    clear_results()

    model_path = os.path.abspath("gesture_recognizer.task")
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handle_result
    )
    recognizer = GestureRecognizer.create_from_options(options)

    cv2.namedWindow("Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error opening webcam")

    # have a 1 loop buffer before recognizing a gesture (clear out old gestures)
    first_loop_flag = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Apply resize and bars
        frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)

        # Print next gesture on screen
        cv2.putText(frame, "Make a closed fist to play game", (off_x + 20, off_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        cv2.putText(frame, "Make a high five to view statistics", (off_x + 20, off_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        cv2.putText(frame, "Make a thumbs down to reset stats", (off_x + 20, off_y + 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        cv2.putText(frame, "WARNING: STAT RESET CANNOT BE REVERTED", (off_x + 20, off_y + 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 2)

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Use video timestamp in ms
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        recognizer.recognize_async(mp_image, timestamp)

        gesture = get_latest_result()
        if not first_loop_flag:
            if gesture == "Closed_Fist":
                return "game"
            elif gesture == "Open_Palm":
                return "stats"
            elif gesture == "Thumb_Down":
                return "reset"
            # Optional: give user feedback on screen
        cv2.imshow("Gesture Recognition", frame)

        first_loop_flag = False

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            return "quit"


def main_loop(difficulty):
    easy_flag = False
    # Handle difficulties
    if difficulty == "easy":
        easy_flag = True
    gesture_list = get_gesture_list(difficulty)
    gesture_dict = get_gesture_dict(difficulty)

    start_time = 0
    end_time = 0
    time_elapsed = 0
    gesture_check = False
    score = 0
    model_path = os.path.abspath("gesture_recognizer.task")
    pick_next_gesture(gesture_list, easy_flag)
    pick_next_gesture(gesture_list, easy_flag)
    print(get_latest_pulled_gesture())
    game_time_start = 0
    first_loop = True
    tbg_list = []
    # pull screen dimensions
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handle_result
    )

    # Create gesture recognizer
    recognizer = GestureRecognizer.create_from_options(options)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error opening webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Apply resize and bars
        frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)

        f_h, f_w = frame.shape[:2]

        # This plays first when starting
        if first_loop:
            first_loop = False
            # ready? Go! logic
            for splash_text in ["Ready?", "Start!"]:
                frame_splash, _, _ = resize_and_pad(cv2.flip(cap.read()[1], 1), screen_width, screen_height)
                frame_splash[:] = (0, 255, 0)
                (tw, th), _ = cv2.getTextSize(splash_text, cv2.FONT_HERSHEY_SIMPLEX, 3 * scale, 3)
                text_x = int((f_w - tw) / 2)
                text_y = int((f_h + th) / 2)

                cv2.putText(frame_splash, splash_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 3 * scale, (0, 0, 0), 3)

                cv2.imshow("Gesture Recognition", frame_splash)
                cv2.waitKey(400)  # Show green for x ms
            # Start timers after message is shown
            game_time_start = time.time()
            start_time = time.time()

        box_x, box_y = off_x + 10, off_y + 10
        box_w, box_h = int(450 * scale), int(120 * scale)

        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (128, 128, 128), -1)

        game_time_elapsed = time_diff(time.time(), game_time_start)

        # Print next gesture on screen
        current_gesture = gesture_dict[get_latest_pulled_gesture()]
        cv2.putText(frame, current_gesture, (box_x + 10, box_y + int(40 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        stats_text = f"{round(time_elapsed, 2)}s | Timer: {int(60 - game_time_elapsed)}"

        cv2.putText(frame, stats_text, (box_x + 10, box_y + int(90 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), 2)

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Use video timestamp in ms
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        recognizer.recognize_async(mp_image, timestamp)

        if gesture_check is True:
            print("Correct!")
            score += 1
            gesture_check = False
            # Grab time elapsed between being shown the gesture and successfully making it
            time_elapsed = time_diff(end_time, start_time)
            print("Time elapsed: " + str(time_diff(end_time, start_time)))
            # Add tbg to list
            tbg_list.append(time_elapsed)
            celebration = random.choice(celebration_list)

            (tw, th), _ = cv2.getTextSize(celebration, cv2.FONT_HERSHEY_SIMPLEX, 2 * scale, 2)
            # Flash screen green

            new_h, new_w = int((f_h - 2 * off_y)), int((f_w - 2 * off_x))
            frame[off_y: off_y + new_h, off_x: off_x + new_w] = (0, 255, 0)

            cv2.putText(frame, celebration, (int((f_w - tw) / 2), int((f_h - th) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2 * scale, (0, 0, 0), 2)
            cv2.imshow("Gesture Recognition", frame)
            cv2.waitKey(400)  # Show green for x ms

            pick_next_gesture(gesture_list, easy_flag)
            print(get_latest_pulled_gesture())
            start_time = time.time()

        gesture_check, end_time = check_gesture()

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

        if int(game_time_elapsed) >= 60:
            break
    append_to_tbg_avg_csv(tbg_list, difficulty)
    append_to_score_csv(score, difficulty)
    cap.release()
    cv2.destroyAllWindows()


def pick_difficulty():
    model_path = os.path.abspath("gesture_recognizer.task")
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handle_result
    )
    recognizer = GestureRecognizer.create_from_options(options)

    cv2.namedWindow("Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error opening webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Apply resize and bars
        frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)

        # Print next gesture on screen
        cv2.putText(frame, "Open Palm: Easy Difficulty", (off_x + 20, off_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        cv2.putText(frame, "Point upwards: Medium Difficulty", (off_x + 20, off_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        cv2.putText(frame, "Thumbs Up: Hard Difficulty", (off_x + 20, off_y + 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Use video timestamp in ms
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        recognizer.recognize_async(mp_image, timestamp)

        if get_latest_result() == "Open_Palm":
            return "easy"
        elif get_latest_result() == "Pointing_Up":
            return "med"
        elif get_latest_result() == "Thumb_Up":
            return "hard"

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break


def confirm_delete():
    model_path = os.path.abspath("gesture_recognizer.task")
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handle_result
    )
    recognizer = GestureRecognizer.create_from_options(options)

    cv2.namedWindow("Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error opening webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Apply resize and bars
        frame, scale, (off_x, off_y) = resize_and_pad(frame, screen_width, screen_height)

        f_h, f_w = frame.shape[:2]

        # Print next gesture on screen
        cv2.putText(frame, "Thumbs Up: CONFIRM - RESET STATS", (off_x + 20, off_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)
        cv2.putText(frame, "Closed Fist: CANCEL", (off_x + 20, off_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, (255, 255, 255), 2)

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Use video timestamp in ms
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        recognizer.recognize_async(mp_image, timestamp)

        if get_latest_result() == "Closed_Fist":
            break
        elif get_latest_result() == "Thumb_Up":
            confirmation = "Stats RESET!!"
            (tw, th), _ = cv2.getTextSize(confirmation, cv2.FONT_HERSHEY_SIMPLEX, 2 * scale, 2)
            # Flash screen green
            new_h, new_w = int((f_h - 2 * off_y)), int((f_w - 2 * off_x))
            frame[off_y: off_y + new_h, off_x: off_x + new_w] = (0, 255, 0)
            cv2.putText(frame, confirmation, (int((f_w - tw) / 2), int((f_h - th) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2 * scale, (0, 0, 0), 2)
            cv2.imshow("Gesture Recognition", frame)
            cv2.waitKey(400)  # Show green for x ms
            delete_csv(score=True)
            break
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break


def driver():
    cv2.destroyAllWindows()
    player_choice = start_screen()
    if player_choice == "game":
        difficulty = pick_difficulty()
        main_loop(difficulty)
        display_graph_screen()
        cv2.waitKey(1)
        time.sleep(0.5)
        return 1
    elif player_choice == "stats":
        display_graph_screen()
        cv2.waitKey(1)
        time.sleep(0.5)
        return 1
    elif player_choice == "reset":
        confirm_delete()
        return 1
    elif player_choice == "quit":
        return 0


if __name__ == "__main__":
    while True:
        result = driver()
        if result == 0:
            break
