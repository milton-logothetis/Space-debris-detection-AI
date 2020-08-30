# Space debris detection AI

This repo hosts all the code and data needed to replicate the AI detection system that was built for my MSc project: "AI for space debris detection & orbit reconstruction".

# Final system example
The following gif visualizes the AI detection stage in a few steps:
1. Records 8 frames of captured video.
2. Pre-processes frames using backround subtraction and image blending.
3. Runs through Faster R-CNN object detector.
4. Collects predictions and plots bounding box (slighlty offset from actual, for visualization purposes).
5. Collects telemetry data, camera parameters and prediction results to calculate parameters of interest (e.g. debris size, speed etc.).
6. Repeats for the next 8 frames.

![Final detection system example](https://github.com/milton-logothetis/Space-debris-detection-AI/blob/master/system_example.gif)
