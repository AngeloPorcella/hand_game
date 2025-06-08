import cv2
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random
import time
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


def main_loop():
    # TODO Track score
    # TODO Track average time between gestures
    # TODO Append files to store data
    # TODO Solve startup bug
    # TODO Build up start screen and end screen
    # TODO Build up graphing component on start menu
    start_time = time.time()
    end_time = 0
    time_elapsed = 0
    gesture_check = False
    score = 0
    model_path = os.path.abspath("gesture_recognizer.task")
    pick_next_gesture()
    print(get_latest_pulled_gesture())
    game_time_start = time.time()
    game_time_elapsed = 0



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
            celebration = random.choice(celebration_list)
            (text_width, text_height), _ = cv2.getTextSize(celebration, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            # Flash screen green
            frame[:] = (0, 255, 0)
            cv2.putText(frame, celebration, (int((frame_width - text_width)/2), int((frame_height - text_height)/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
