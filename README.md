## Overview
This project consists of two applications:

#Hand_game.py:
A simple game created using OpenCV and MediaPipe.
This is an AR game similar to what you would find on the XBOX Kinect.
Hand tracking via the MP library allows you to "hit" randomly spawned shapes using your webcam.
Watch out, each shape can de-spawn after a set period.

#Rehab.py:
Following my father's stroke i'm pivoting this project to stroke recovery.
A gesture recgonition game using OpenCV and MediaPipe.
The user replicates the hand signals displayed and gains score.
Time between gestures is recorded and graphed for a tangible metric of progress.


## Setup

I reccomend creating a virtual environment to install dependencies.

Requires Numpy, pandas, OpenCV, and MediaPipe

Run the commands:
```
pip install opencv-python
pip install mediapipe
pip install numpy
```
Made by Angelo
