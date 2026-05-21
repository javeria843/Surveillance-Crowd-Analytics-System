# Smart Surveillance & Crowd Analytics System

---

## Introduction

The **Smart Surveillance & Crowd Analytics System** is a real-time computer vision project developed using Python, OpenCV, and YOLOv8. The system is designed to monitor crowd activity from video footage, detect and track individuals, analyze crowd density inside a restricted area, and generate alerts whenever overcrowding occurs.

In addition, the system creates movement heatmaps, performs edge detection analysis, and produces detailed performance metrics for surveillance monitoring and crowd behavior analysis.

---

## Main Features

* **Real-time Human Detection** using YOLOv8
* **Object Tracking** with unique track IDs
* **Restricted Zone Monitoring**
* **Crowd Density Alert System**
* **Movement Heatmap Generation**
* **Edge Detection Sampling** using Canny and Sobel operators
* **Performance Metrics Reporting**
* **Annotated Video Generation**
* **Crowd Density Graph Visualization**

---

# Project Structure

```text
.
├── main.py                  # Main Python file
├── crowd_video2.mp4         # Input surveillance video
├── yolov8n.pt               # YOLOv8 model weights
├── output_annotated.avi     # Generated annotated output video
├── density_graph.png        # Crowd density graph
└── edge_samples.png         # Edge detection sample results
```

---

# Required Libraries

Install the required dependencies using:

```bash
pip install opencv-python numpy ultralytics matplotlib
```

| Library     | Purpose                                         |
| ----------- | ----------------------------------------------- |
| OpenCV      | Video processing and computer vision operations |
| NumPy       | Numerical and array operations                  |
| Ultralytics | YOLOv8 object detection and tracking            |
| Matplotlib  | Graph plotting and visualization                |

> The YOLOv8 model (`yolov8n.pt`) is automatically downloaded during the first execution if it is not already available.

---

# System Configuration

The following parameters can be modified inside `main.py`:

| Parameter           | Description                                       |
| ------------------- | ------------------------------------------------- |
| `VIDEO_PATH`        | Path of the input video                           |
| `OUTPUT_PATH`       | Path of the output annotated video                |
| `DENSITY_THRESHOLD` | Maximum allowed people inside the restricted zone |
| `TRAIL_LENGTH`      | Length of tracking motion trails                  |
| `SAMPLE_INTERVAL`   | Interval for edge detection sampling              |
| `MAX_SAMPLES`       | Maximum saved edge samples                        |
| `SKIP_FRAMES`       | Number of skipped frames for faster processing    |

---

# Working Procedure

1. The input surveillance video is loaded using OpenCV.
2. YOLOv8 detects people in each frame.
3. ByteTrack tracking assigns unique IDs to detected individuals.
4. A restricted polygon zone is monitored continuously.
5. The system counts people entering the zone.
6. If the crowd count exceeds the threshold value, an alert is triggered.
7. Movement positions are accumulated to generate a heatmap.
8. Edge detection samples are collected periodically using:

   * Canny Edge Detection
   * Sobel Edge Detection
9. Processed frames are saved into an annotated output video.
10. Density graphs and analysis reports are generated after processing.

---

# Output Description

## 1. Annotated Video

The generated output video contains:

* Bounding boxes around detected individuals
* Green boxes for people outside the zone
* Red boxes for people inside the restricted zone
* Track ID and confidence score labels
* Motion tracking trails
* Heatmap overlay visualization
* Live crowd statistics panel
* Crowd alert banner when density exceeds the threshold

---

## 2. Density Graph

The system generates a graph showing crowd density variations over time throughout the video.

---

## 3. Edge Detection Samples

Edge detection images generated using:

* Canny Operator
* Sobel Operator

These samples help analyze object boundaries and movement patterns.

---

# Performance Metrics

After execution, the system prints a detailed analytics report including:

## Detection Metrics

* Average confidence score
* Minimum and maximum confidence
* Average detections per frame
* Processing FPS
* Average frame processing time

## Tracking Metrics

* Total unique persons tracked
* Average tracking duration
* Tracking stability analysis
* Estimated ID switch count

---

# Computer Vision Techniques Used

| Technique            | Purpose                      |
| -------------------- | ---------------------------- |
| YOLOv8 Detection     | Human detection              |
| ByteTrack Tracking   | Multi-object tracking        |
| Canny Edge Detection | Edge extraction              |
| Sobel Operator       | Gradient-based edge analysis |
| Heatmap Generation   | Crowd movement visualization |
| Polygon Zone Testing | Restricted area monitoring   |
| Video Annotation     | Real-time visualization      |

---

# Important Notes

* The restricted zone is dynamically adjusted according to video resolution.
* Detections below the confidence threshold of **0.35** are ignored.
* Heatmaps are normalized before visualization for better clarity.
* Frame skipping is implemented to improve processing speed while maintaining acceptable accuracy.

---

# Conclusion

The Smart Surveillance & Crowd Analytics System demonstrates the practical implementation of modern computer vision techniques for intelligent surveillance applications. By combining real-time detection, tracking, crowd monitoring, and analytical visualization, the system provides an efficient solution for monitoring crowded environments and detecting abnormal crowd behavior automatically.

---

# Course Information

**Course:** Computer Vision

## Group Members

| Name          | Roll Number |
| ------------- | ----------- |
| Mehak Faheem  | 23-AI-47    |
| Arekah Taj    | 23-AI-21    |
| Javeria Iqbal | 23-AI-45    |
