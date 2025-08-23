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

celebration_list = ["FUCK YEAH!",
                    "Yay!",
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
# available gestures and common names
gesture_dict = {"Closed_Fist": "Closed Fist",
                "Open_Palm": "High Five",
                "Pointing_Up": "Point Up",
                "Thumb_Down": "Thumbs Down",
                "Thumb_Up": "Thumbs Up",
                "Victory": "Peace Sign",
                "ILoveYou": "I Love You"}
gesture_dict_easy = {"Closed_Fist": "Closed Fist",
                     "Open_Palm": "High Five",
                     "Pointing_Up": "Point Up"}
gesture_list = list(gesture_dict.keys())
informal_gesture_list_names = list(gesture_dict.values())


def append_to_tbg_avg_csv(data_list):
    avg_session_tbg = average_list(data_list)
    with open('avg_tbg.csv', 'a') as file:
        file.write(str(avg_session_tbg) + "\n")  # newline instead of comma+space


def append_to_total(data_point):
    print("This is where I will append to a file for future graphing")


# Average datapoints
def average_list(data_list):
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
    if size >= 2:
        pulled_gestures.pop(0)
    pulled_gestures.append(item)


# get the most recently recognized gesture
def get_latest_result():
    if len(results) == 0:
        return "empty"
    return results[-1]


def get_latest_pulled_gesture():
    if len(pulled_gestures) == 0:
        return "empty"
    return pulled_gestures[0]


def pick_next_gesture():
    chosen_gesture = random.choice(gesture_list)
    # If the chosen gesture has been performed recently, keep trying until a new choice is chosen
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


def create_graph_from_csv(csv_path):
    # Read CSV with no header (just a single column of numbers)
    try:
        df = pd.read_csv(csv_path, header=None, names=['avg_tbg'])
    except FileNotFoundError:
        print("No data, play a game first.")
        return None

    if df.empty:
        print("No data in csv, play a game first.")
        return None

    # Extract y-values (averages) and x-values (index/attempt number)
    y_values = df['avg_tbg'].astype(float).values
    x_values = np.arange(1, len(y_values) + 1)

    # Make figure
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_title("Average Time Between Gestures")
    ax.set_xlabel("Session")
    ax.set_ylabel("Avg Time")

    # Plot line with points
    ax.plot(x_values, y_values, marker='o', color='blue')

    plt.show()

    # Attach canvas to figure and draw
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Convert to NumPy array
    width, height = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = buf.reshape((height, width, 4))  # RGBA image

    # Convert RGBA to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image_bgr


def delete_csv(file_path="avg_tbg.csv"):
    """Delete the CSV file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} deleted (stats reset).")
    else:
        print(f"{file_path} not found (nothing to reset).")

def start_screen():

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
        frame_height, frame_width = frame.shape[:2]

        frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
        frame_height, frame_width = frame.shape[:2]  # update after resize

        box_x, box_y = int(0.01 * frame_width), int(0.01 * frame_height)  # relative positioning
        box_width, box_height = int(0.3 * frame_width), int(0.1 * frame_height)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (128, 128, 128), -1)

        # Print next gesture on screen
        cv2.putText(frame, "Make a closed fist to start game", (int(10), int(30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Make a high five to view statistics", (int(10), int(75)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Make a thumbs down to reset stats! WARNING: CANNOT BE REVERTED", (int(10), int(120)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Use video timestamp in ms
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        recognizer.recognize_async(mp_image, timestamp)

        if get_latest_result() == "Closed_Fist":
            return "game"
        elif get_latest_result() == "Open_Palm":
            return "stats"
        elif get_latest_result() == "Thumb_Down":
            delete_csv("avg_tbg.csv")
            # Optional: give user feedback on screen
            cv2.putText(frame, "Stats Reset!",
                        (int(0.4 * frame_width), int(0.5 * frame_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow("Gesture Recognition", frame)
            cv2.waitKey(800)  # show message for a moment
            return "reset"

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

def main_loop():
    start_time = 0
    end_time = 0
    time_elapsed = 0
    gesture_check = False
    score = 0
    model_path = os.path.abspath("gesture_recognizer.task")
    pick_next_gesture()
    pick_next_gesture()
    print(get_latest_pulled_gesture())
    game_time_start = 0
    first_loop = True
    tbg_list = []

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
        frame_height, frame_width = frame.shape[:2]

        # This plays first when starting
        if first_loop:
            # Only enter the loop once
            first_loop = False
            # Flash the screen green and give a message
            splash_text = "Ready?"
            (first_text_width, first_text_height), _ = cv2.getTextSize(splash_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            frame[:] = (0, 255, 0)
            cv2.putText(frame, splash_text, (int((frame_width - first_text_width)/2),
                                             int((frame_height - first_text_height)/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
            cv2.imshow("Gesture Recognition", frame)
            cv2.waitKey(400)  # Show message for x ms
            splash_text = "Start!"
            (first_text_width, first_text_height), _ = cv2.getTextSize(splash_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            frame[:] = (0, 255, 0)
            cv2.putText(frame, splash_text, (int((frame_width - first_text_width)/2),
                                             int((frame_height - first_text_height)/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
            cv2.imshow("Gesture Recognition", frame)
            cv2.waitKey(400)  # Show green for x ms
            # Start timers after message is shown
            game_time_start = time.time()
            start_time = time.time()

        box_x, box_y = 1, 1
        box_width, box_height = 400, 100

        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width,  box_y + box_height), (128, 128, 128), -1)

        game_time_elapsed = time_diff(time.time(), game_time_start)

        # Print next gesture on screen
        cv2.putText(frame, gesture_dict[get_latest_pulled_gesture()], (int(10), int(30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(round(time_elapsed, 3)) + " Seconds", (int(10), int(75)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(int(60 - game_time_elapsed)), (int(350), int(75)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
            (text_width, text_height), _ = cv2.getTextSize(celebration, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            # Flash screen green
            frame[:] = (0, 255, 0)
            cv2.putText(frame, celebration, (int((frame_width - text_width)/2), int((frame_height - text_height)/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
            cv2.imshow("Gesture Recognition", frame)
            cv2.waitKey(400)  # Show green for x ms

            pick_next_gesture()
            print(get_latest_pulled_gesture())
            start_time = time.time()

        gesture_check, end_time = check_gesture()

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

        if int(game_time_elapsed) >= 60:
            break
    append_to_tbg_avg_csv(tbg_list)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    player_choice = start_screen()
    if player_choice == "game":
        main_loop()
    elif player_choice == "stats":
        create_graph_from_csv("avg_tbg.csv")
        print("stats!")
        #stats_screen()
    elif player_choice == "reset":
        print("reset!")

