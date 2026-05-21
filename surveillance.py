# Smart Surveillance & Crowd Analytics System
# 23-AI-21, 23-AI-45, 23-AI-47

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')                      # prevents display conflicts on Windows
import matplotlib.pyplot as plt

VIDEO_PATH        = "crowd_video2.mp4"
OUTPUT_PATH       = "output_annotated.avi"
DENSITY_THRESHOLD = 5
TRAIL_LENGTH      = 25
SAMPLE_INTERVAL   = 60
MAX_SAMPLES       = 5
SKIP_FRAMES       = 3    # run YOLO every 3rd frame — speeds up preview

#  LOAD YOLO MODEL
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded!\n")

def get_centroid(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def is_inside_zone(point, zone_polygon):
    result = cv2.pointPolygonTest(
        zone_polygon,
        (float(point[0]), float(point[1])),
        False
    )
    return result >= 0

def apply_canny_edges(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return edges

def apply_sobel(frame):
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x   = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y   = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(np.clip(magnitude, 0, 255))

def update_heatmap(heatmap, centroid, sigma=20):
    cv2.circle(heatmap, centroid, sigma, 1, -1)
    return heatmap

#  SAVE EDGE DETECTION SAMPLES
def save_edge_samples(samples):
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (original, edges, sobel) in enumerate(samples):
        axes[i][0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f"Sample {i+1}: Original Frame")
        axes[i][0].axis("off")

        axes[i][1].imshow(edges, cmap="gray")
        axes[i][1].set_title(f"Sample {i+1}: Canny Edge Detection")
        axes[i][1].axis("off")

        axes[i][2].imshow(sobel, cmap="gray")
        axes[i][2].set_title(f"Sample {i+1}: Sobel Gradient")
        axes[i][2].axis("off")

    plt.suptitle("Scene Structure Analysis — Edge Detection Samples",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("edge_samples.png", dpi=150)   # save BEFORE show
    plt.close('all')
    print("Saved → edge_samples.png")

def main():

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {VIDEO_PATH}")
        return

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS)
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Detected FPS: {FPS}")
    print(f"Video info: {W}x{H} @ {FPS} fps | {TOTAL} total frames\n")

    # Resize to 854 width — faster on CPU
    SCALE = 854 / W
    W = 854
    H = int(H * SCALE)

    # Restricted zone — proportional to frame size
    ZONE = np.array([
        [int(W * 0.25), int(H * 0.40)],
        [int(W * 0.75), int(H * 0.40)],
        [int(W * 0.85), int(H * 0.90)],
        [int(W * 0.15), int(H * 0.90)],
    ], dtype=np.int32)

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (W, H))

    # Data structures
    trails      = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    heatmap     = np.zeros((H, W), dtype=np.float32)
    density_log = []
    frame_log   = []
    edge_samples = []
    frame_count  = 0
    last_results = None    # stores last YOLO result for skipped frames

    print("Processing... Press Q to stop early.\n")

    cv2.namedWindow("Crowd Surveillance System", cv2.WINDOW_NORMAL)

    #  MAIN LOOP
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (W, H))

        # Edge detection samples 
        if frame_count % SAMPLE_INTERVAL == 0 and len(edge_samples) < MAX_SAMPLES:
            edges = apply_canny_edges(frame)
            sobel = apply_sobel(frame)
            edge_samples.append((frame.copy(), edges, sobel))
            print(f"  Edge sample {len(edge_samples)} saved at frame {frame_count}")

        # YOLO — only every SKIP_FRAMES frames
        if frame_count % SKIP_FRAMES == 0:
            last_results = model.track(frame, classes=[0], persist=True, verbose=False)
        results = last_results

        # Draw restricted zone 
        zone_overlay = frame.copy()
        cv2.fillPoly(zone_overlay, [ZONE], (0, 220, 220))
        frame = cv2.addWeighted(frame, 0.82, zone_overlay, 0.18, 0)
        cv2.polylines(frame, [ZONE], isClosed=True, color=(0, 200, 200), thickness=2)
        cv2.putText(frame, "RESTRICTED ZONE",
                    (ZONE[0][0], ZONE[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        # Process detections
        people_in_zone = 0
        boxes_data = results[0].boxes if results is not None else None

        if boxes_data is not None and len(boxes_data) > 0:
            for box in boxes_data:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id   = int(box.id[0]) if box.id is not None else -1
                confidence = float(box.conf[0])

                if confidence < 0.35:
                    continue

                centroid = get_centroid(x1, y1, x2, y2)

                if track_id >= 0:
                    trails[track_id].append(centroid)

                heatmap = update_heatmap(heatmap, centroid)

                in_zone   = is_inside_zone(centroid, ZONE)
                box_color = (0, 0, 255) if in_zone else (0, 255, 0)
                if in_zone:
                    people_in_zone += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                label = f"ID:{track_id} {confidence:.2f}" if track_id >= 0 else f"{confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), box_color, -1)
                cv2.putText(frame, label, (x1 + 1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if track_id >= 0:
                    trail_list = list(trails[track_id])
                    for j in range(1, len(trail_list)):
                        thickness = max(1, int((j / len(trail_list)) * 3))
                        cv2.line(frame, trail_list[j-1], trail_list[j], (0, 140, 255), thickness)
                        cv2.circle(frame, trail_list[j], 2, (0, 200, 255), -1)

        # Crowd Alert
        alert_active = people_in_zone > DENSITY_THRESHOLD
        if alert_active:
            alert_overlay = frame.copy()
            cv2.fillPoly(alert_overlay, [ZONE], (0, 0, 255))
            frame = cv2.addWeighted(frame, 0.65, alert_overlay, 0.35, 0)

            alert_text = "!! CROWD ALERT !!"
            (aw, ah), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
            ax = int((W - aw) / 2)       # horizontally centered
            ay = int(H * 0.88)            # near bottom of frame
            # Dark background behind alert text for visibility
            cv2.rectangle(frame, (ax - 10, ay - ah - 10), (ax + aw + 10, ay + 10),
                          (0, 0, 0), -1)
            cv2.putText(frame, alert_text, (ax, ay),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 3)

        # Heatmap overlay
        norm_heat  = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        color_heat = cv2.applyColorMap(np.uint8(norm_heat), cv2.COLORMAP_JET)
        frame      = cv2.addWeighted(frame, 0.75, color_heat, 0.25, 0)

        # Info panel (top-left HUD) 
        total_people = len(boxes_data) if boxes_data is not None else 0
        status_text  = "ALERT!" if alert_active else "Normal"
        info_lines   = [
            f"Frame: {frame_count}/{TOTAL}",
            f"People detected: {total_people}",
            f"In zone: {people_in_zone}",
            f"Threshold: {DENSITY_THRESHOLD}",
            f"Status: {status_text}",
        ]
        cv2.rectangle(frame, (5, 5), (285, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (285, 150), (120, 120, 120), 1)
        for i, line in enumerate(info_lines):
            color = (0, 0, 255) if ("ALERT" in line and alert_active) else (255, 255, 255)
            cv2.putText(frame, line, (12, 28 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Log density 
        density_log.append(people_in_zone)
        frame_log.append(frame_count)

        # Write & display 
        out.write(frame)
        cv2.imshow("Crowd Surveillance System", frame)

        # waitKey delay matches FPS so preview plays at real speed
        if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord("q"):
            print("Stopped early.")
            break

        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}/{TOTAL} | In zone: {people_in_zone}")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Saved → {OUTPUT_PATH}")

    # Density graph
    plt.figure(figsize=(14, 5))
    plt.plot(frame_log, density_log, color="steelblue", linewidth=1.5, label="People in Zone")
    plt.axhline(y=DENSITY_THRESHOLD, color="red", linestyle="--",
                linewidth=1.5, label=f"Alert Threshold = {DENSITY_THRESHOLD}")
    plt.fill_between(frame_log, density_log, alpha=0.25, color="steelblue")
    plt.xlabel("Frame Number")
    plt.ylabel("Number of People in Zone")
    plt.title("Crowd Density Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("density_graph.png", dpi=150)
    plt.close('all')                           # close before opening edge samples
    print("Density graph saved → density_graph.png")

    # Edge samples
    if edge_samples:
        save_edge_samples(edge_samples)
    else:
        print("No edge samples collected.")

if __name__ == "__main__":
    main()