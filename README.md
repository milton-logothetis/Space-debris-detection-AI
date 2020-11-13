# Space debris detection AI

This repo hosts all the code and data needed to replicate the AI detection system that was built for my MSc project: "AI for space debris detection & orbit reconstruction".

The data can be found in the following Google Drive link: https://drive.google.com/drive/folders/1fwEwQbzbYTEcPcD9p_AiPfQqSwP30uyv?usp=sharing

The complete Thesis report can be found at:
https://drive.google.com/file/d/1lmV6VItuXquvpM_5oipap9nvMzqXNt8k/view?usp=sharing

# Final system example
The following gif visualizes the AI detection stage in a few steps:
1. Records 8 frames of captured video.
2. Pre-processes frames using backround subtraction and image blending.
3. Runs through Faster R-CNN object detector.
4. Collects predictions and plots bounding box (slighlty offset from actual, for visualization purposes).
5. Centers camera payload to debris origin (assuming an appropriate gimbal system).
6. Collects telemetry data, camera parameters and prediction results to calculate & display parameters of interest (e.g. debris position, speed etc.).
7. Repeats for the next 8 frames.

![Final detection system example](https://github.com/milton-logothetis/Space-debris-detection-AI/blob/master/system_example.gif)

*note the actual speed of the process has been slowed down to around 1/3 through .gif conversion
