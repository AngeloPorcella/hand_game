## Why I made this
This project is a personal one, my father suffered a stroke in the spring of 2025. In his mid 60's, and in great health it came as a completely unexpected event. During the following days I spent in the hospital with him and my family, I kept busy working on this program as some way to wrestle control out of an upsetting situation. Overall the goal of this is to provide a free, easily accessible, rehabilitation tool that hopefully can make a difference for those battling with motor function recovery. I hope to get this in the hands of some professionals and get some feedback soon.

## Overview
This project consists of two applications:

Hand_game.py:
A simple game created using OpenCV and MediaPipe.
This is an AR game similar to what you would find on the XBOX Kinect.
Hand tracking via the MP library allows you to "hit" randomly spawned shapes using your webcam.
Watch out, each shape can de-spawn after a set period.
Features two different difficulties with differing spawn rates and win conditions.
Stats are recorded and displayed to visualize motor-skill improvement.
Focus for this game is on shoulder and arm dexterity.

Rehab.py:
A gesture recgonition game using OpenCV and MediaPipe.
The user replicates the hand signals displayed and gains score.
Features three different difficulties that encompass various gesture sets ranging in complexity.
Stats are recorded and displayed to visualize motor-skill improvement.
Focus for this game is on finger and wrist mobility.


## Setup

I reccomend creating a virtual environment to install dependencies.

Requires Numpy, pandas, OpenCV, and MediaPipe

Run the commands:
```
pip install opencv-python
pip install mediapipe
pip install numpy
```
Made by Angelo Porcella
