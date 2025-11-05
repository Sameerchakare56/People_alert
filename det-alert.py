# main.py - Optimized FastAPI Person Tracking System with Side Detection
# Supports left/right side boundary crossing detection

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import requests
from datetime import datetime
import base64
from typing import Dict, Set, Optional
import json
from collections import defaultdict
import asyncio
import torch
import os
import time
import math

app = FastAPI(title="Person Tracking System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration with Side Detection Support
class Config:
    # Video Source Configuration
    VIDEO_SOURCE_TYPE = "local"
    
    RTSP_URL = "http://192.168.1.12:5000/live_akash"
    LOCAL_VIDEO_PATH = "vi.mp4"
    #RTSP_URL = "rtsp://admin:tdbtech4189@192.168.1.250:554/Streaming/Channels/102"
    #VIDEO_SOURCE_TYPE = "rtsp"
    #RTSP_URL = "rtsp://admin:tdbtech4189@192.168.1.250:554"
    #RTSP_URL = 0
    # API Configuration
    API_ENDPOINT = "http://localhost:8000/receive_alert"

    # BALANCED PERFORMANCE SETTINGS
    PROCESS_WIDTH = 640
    PROCESS_HEIGHT = 480

    # Stream output settings
    STREAM_WIDTH = 640
    STREAM_HEIGHT = 480
    STREAM_FPS = 15
    JPEG_QUALITY = 85

    # Skip frames for detection
    DETECTION_FRAME_SKIP = 1

    # ANGLED BOUNDARY LINE SETTINGS
    BOUNDARY_LINE_RATIO = 0.5
    BOUNDARY_ANGLE = 0
    BOUNDARY_ORIENTATION = "horizontal"

    # NEW: Side Detection Settings
    DETECTION_SIDE = "both"  # Options: "left", "right", "both"

    # Line equation parameters
    LINE_POINT1 = None
    LINE_POINT2 = None

    # Model Configuration
    MODEL_PATH = "yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    MAX_DISAPPEARED = 30

    USE_GPU = torch.cuda.is_available()

    # Image compression for alerts
    ALERT_IMAGE_WIDTH = 300
    ALERT_IMAGE_HEIGHT = 400
    ALERT_JPEG_QUALITY = 75

    # Video Buffer Settings
    BUFFER_SIZE = 1
    LOOP_VIDEO = True


config = Config()


def calculate_angled_boundary_line(frame_height, frame_width, angle, position_ratio):
    """Calculate boundary line endpoints based on angle and position"""
    angle_rad = math.radians(angle)

    if angle == 0 or angle == 180:
        y_pos = int(frame_height * position_ratio)
        point1 = (0, y_pos)
        point2 = (frame_width, y_pos)
    elif angle == 90 or angle == 270:
        x_pos = int(frame_width * position_ratio)
        point1 = (x_pos, 0)
        point2 = (x_pos, frame_height)
    else:
        center_x = frame_width / 2
        center_y = frame_height / 2

        perpendicular_angle = angle_rad + math.pi / 2
        offset_distance = (position_ratio - 0.5) * max(frame_width, frame_height)

        center_x += offset_distance * math.cos(perpendicular_angle)
        center_y += offset_distance * math.sin(perpendicular_angle)

        line_length = max(frame_width, frame_height) * 2

        dx = line_length * math.cos(angle_rad)
        dy = line_length * math.sin(angle_rad)

        point1 = (int(center_x - dx / 2), int(center_y - dy / 2))
        point2 = (int(center_x + dx / 2), int(center_y + dy / 2))

    return point1, point2


def point_to_line_distance_signed(point, line_point1, line_point2):
    """
    Calculate signed distance from point to line
    Positive: one side (right/top), Negative: other side (left/bottom)
    """
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2

    numerator = (y2 - y1) * x0 - (x2 - x1) * y0 + (x2 - x1) * y1 - (y2 - y1) * x1
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    if denominator == 0:
        return 0

    return numerator / denominator


def get_crossing_direction(prev_distance, current_distance):
    """
    Determine crossing direction
    Returns: "left_to_right" or "right_to_left" or None
    """
    if prev_distance < 0 and current_distance > 0:
        return "left_to_right"
    elif prev_distance > 0 and current_distance < 0:
        return "right_to_left"
    return None


# Person tracker class
class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracked_persons = {}
        self.crossed_ids: Set[int] = set()

    def register(self, centroid, bbox):
        person_id = self.next_id
        self.tracked_persons[person_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "disappeared": 0,
            "crossed": False,
            "prev_centroid": centroid,
            "prev_line_distance": None,
            "crossing_direction": None
        }
        self.next_id += 1
        return person_id

    def deregister(self, person_id):
        if person_id in self.tracked_persons:
            del self.tracked_persons[person_id]
            if person_id in self.crossed_ids:
                self.crossed_ids.remove(person_id)

    def update(self, detections):
        if len(detections) == 0:
            for person_id in list(self.tracked_persons.keys()):
                self.tracked_persons[person_id]["disappeared"] += 1
                if self.tracked_persons[person_id]["disappeared"] > config.MAX_DISAPPEARED:
                    self.deregister(person_id)
            return self.tracked_persons

        input_centroids = []
        input_bboxes = []
        for det in detections:
            x1, y1, x2, y2 = det
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
            input_bboxes.append((x1, y1, x2, y2))

        if len(self.tracked_persons) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            person_ids = list(self.tracked_persons.keys())
            object_centroids = [self.tracked_persons[pid]["centroid"] for pid in person_ids]

            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = np.linalg.norm(
                        np.array(object_centroids[i]) - np.array(input_centroids[j])
                    )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 50:
                    continue

                person_id = person_ids[row]
                self.tracked_persons[person_id]["prev_centroid"] = self.tracked_persons[person_id]["centroid"]
                self.tracked_persons[person_id]["centroid"] = input_centroids[col]
                self.tracked_persons[person_id]["bbox"] = input_bboxes[col]
                self.tracked_persons[person_id]["disappeared"] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])

            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                person_id = person_ids[row]
                self.tracked_persons[person_id]["disappeared"] += 1
                if self.tracked_persons[person_id]["disappeared"] > config.MAX_DISAPPEARED:
                    self.deregister(person_id)

        return self.tracked_persons


# Global objects
model = None
tracker = PersonTracker()
cap = None
latest_crossing = None
video_source_info = {}
frame_count = 0


def initialize_model():
    """Initialize YOLO model with optimization"""
    global model
    try:
        original_load = torch.load

        def patched_load(f, *args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(f, *args, **kwargs)

        torch.load = patched_load
        model = YOLO(config.MODEL_PATH)

        if config.USE_GPU:
            model.to('cuda')
            print("✅ YOLO model loaded on GPU")
        else:
            print("✅ YOLO model loaded on CPU")

        torch.load = original_load

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def initialize_video_source():
    """Initialize video source"""
    global cap, video_source_info

    try:
        source_type = config.VIDEO_SOURCE_TYPE.lower()

        if source_type == "rtsp":
            print(f"🎥 Connecting to RTSP stream: {config.RTSP_URL}")
            cap = cv2.VideoCapture(config.RTSP_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.BUFFER_SIZE)
            video_source_info = {"type": "rtsp", "source": config.RTSP_URL}

        elif source_type == "local":
            if not os.path.exists(config.LOCAL_VIDEO_PATH):
                print(f"❌ Local video file not found: {config.LOCAL_VIDEO_PATH}")
                return False
            cap = cv2.VideoCapture(config.LOCAL_VIDEO_PATH)
            video_source_info = {"type": "local", "source": config.LOCAL_VIDEO_PATH}

        elif source_type == "webcam":
            cap = cv2.VideoCapture(0)
            video_source_info = {"type": "webcam", "source": "device_0"}

        if not cap.isOpened():
            print(f"⚠️ Failed to open {source_type} source")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Video source initialized: {width}x{height}")

        video_source_info.update({"width": width, "height": height})
        return True

    except Exception as e:
        print(f"❌ Error initializing video source: {e}")
        return False


def compress_image(image, max_width, max_height, quality):
    """Compress and resize image for optimization"""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    return buffer


def send_alert(person_id: int, person_image: np.ndarray, timestamp: str,
               frame_height: int, crossing_direction: str):
    """Send alert with compressed image and crossing direction"""
    global latest_crossing

    try:
        buffer = compress_image(
            person_image,
            config.ALERT_IMAGE_WIDTH,
            config.ALERT_IMAGE_HEIGHT,
            config.ALERT_JPEG_QUALITY
        )
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "person_id": person_id,
            "timestamp": timestamp,
            "image": img_base64,
            "alert_type": "angled_boundary_crossed",
            "boundary_angle": config.BOUNDARY_ANGLE,
            "boundary_ratio": config.BOUNDARY_LINE_RATIO,
            "line_point1": config.LINE_POINT1,
            "line_point2": config.LINE_POINT2,
            "crossing_direction": crossing_direction,
            "detection_side": config.DETECTION_SIDE
        }

        latest_crossing = payload

        import threading
        def send_request():
            try:
                requests.post(config.API_ENDPOINT, json=payload, timeout=5)
            except:
                pass

        thread = threading.Thread(target=send_request, daemon=True)
        thread.start()

    except Exception as e:
        print(f"❌ Error preparing alert: {e}")


def check_boundary_crossing(person_id: int, person_data: dict, frame: np.ndarray,
                            line_point1: tuple, line_point2: tuple):
    """Check if person crossed the boundary line based on detection side setting"""
    current_centroid = person_data["centroid"]
    prev_centroid = person_data["prev_centroid"]
    bbox = person_data["bbox"]

    # Calculate signed distance from line
    current_distance = point_to_line_distance_signed(current_centroid, line_point1, line_point2)
    prev_distance = point_to_line_distance_signed(prev_centroid, line_point1, line_point2)

    # Determine crossing direction
    crossing_direction = get_crossing_direction(prev_distance, current_distance)

    # Store current distance for next frame
    person_data["prev_line_distance"] = current_distance

    # Check if person crossed and if it matches the detection side setting
    if crossing_direction and person_id not in tracker.crossed_ids:
        detection_side = config.DETECTION_SIDE.lower()

        should_alert = False

        if detection_side == "both":
            should_alert = True
        elif detection_side == "left" and crossing_direction == "left_to_right":
            should_alert = True
        elif detection_side == "right" and crossing_direction == "right_to_left":
            should_alert = True

        if should_alert:
            tracker.crossed_ids.add(person_id)
            person_data["crossing_direction"] = crossing_direction
            timestamp = datetime.now().isoformat()

            x1, y1, x2, y2 = bbox
            person_img = frame[int(y1):int(y2), int(x1):int(x2)]

            if person_img.size > 0:
                send_alert(person_id, person_img, timestamp, frame.shape[0], crossing_direction)
                print(
                    f"🚨 ALERT: Person {person_id} crossed boundary ({crossing_direction}) - Detection side: {detection_side}")

            return True
    return False


def draw_person_path(frame, person_id, centroid, path_history, color, max_length=30):
    """
    🟢 Draws and updates the motion path (trajectory) of a tracked person.
    Args:
        frame: Current video frame (numpy array)
        person_id: Unique tracker ID for the person
        centroid: Current (x, y) centroid of the person
        path_history: Dictionary storing list of previous centroids for each ID
        color: Color of the path (tuple: BGR)
        max_length: Maximum number of past points to retain in the trajectory
    """
    # Initialize path if first time
    if person_id not in path_history:
        path_history[person_id] = []

    # Append current centroid to path
    path_history[person_id].append(centroid)

    # Limit path length
    if len(path_history[person_id]) > max_length:
        path_history[person_id].pop(0)

    # Draw the trajectory line
    pts = path_history[person_id]
    for j in range(1, len(pts)):
        if pts[j - 1] is None or pts[j] is None:
            continue
        cv2.line(frame, pts[j - 1], pts[j], color, 5)

    return path_history

def generate_frames():
    """Frame generation with side detection visualization"""
    global cap, model, tracker, frame_count

    last_detections = []
    path_history = {}
    while True:
        success, frame = cap.read()

        if not success:
            if video_source_info.get("type") == "local" and config.LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        frame_count += 1
        frame_height, frame_width = frame.shape[:2]

        # Resize frame for processing
        process_frame = cv2.resize(frame, (config.PROCESS_WIDTH, config.PROCESS_HEIGHT))
        scale_x = frame_width / config.PROCESS_WIDTH
        scale_y = frame_height / config.PROCESS_HEIGHT

        # Run detection
        if frame_count % config.DETECTION_FRAME_SKIP == 0:
            results = model(process_frame, conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append((
                            x1 * scale_x, y1 * scale_y,
                            x2 * scale_x, y2 * scale_y
                        ))
            last_detections = detections
        else:
            detections = last_detections

        # Update tracker
        tracked_persons = tracker.update(detections)

        # Calculate boundary line
        line_point1, line_point2 = calculate_angled_boundary_line(
            frame_height, frame_width,
            config.BOUNDARY_ANGLE,
            config.BOUNDARY_LINE_RATIO
        )
        config.LINE_POINT1 = line_point1
        config.LINE_POINT2 = line_point2

        # Draw boundary line
        cv2.line(frame, line_point1, line_point2, (0, 255, 0), 3)

        # Draw detection side indicators
        mid_x = (line_point1[0] + line_point2[0]) // 2
        mid_y = (line_point1[1] + line_point2[1]) // 2

        # Main label
        label_text = f"Angle: {config.BOUNDARY_ANGLE}° | Pos: {int(config.BOUNDARY_LINE_RATIO * 100)}%"
        cv2.putText(frame, label_text, (mid_x - 100, mid_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detection side label
        side_text = f"Detection: {config.DETECTION_SIDE.upper()}"
        side_color = (0, 255, 255) if config.DETECTION_SIDE == "both" else (255, 165, 0)
        cv2.putText(frame, side_text, (mid_x - 80, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, side_color, 2)

        # Draw tracked persons
        for person_id, person_data in tracked_persons.items():
            bbox = person_data["bbox"]
            x1, y1, x2, y2 = bbox
            
            centroid = person_data.get("centroid", (int((x1 + x2) / 2), int((y1 + y2) / 2))) 

            check_boundary_crossing(person_id, person_data, frame, line_point1, line_point2)

            color = (0, 0, 255) if person_id in tracker.crossed_ids else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"ID: {person_id}"
            if person_id in tracker.crossed_ids:
                direction = person_data.get("crossing_direction", "")
                direction_symbol = "→" if direction == "left_to_right" else "←"
                label += f" [CROSSED {direction_symbol}]"

            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            path_history = draw_person_path(frame, person_id, centroid, path_history, color, max_length=500)  
        # Info overlay
        info_text = f"Tracked: {len(tracked_persons)} | Alerts: {len(tracker.crossed_ids)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Compress output frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Person Tracking System with Side Detection...")
    print(f"⚡ GPU Acceleration: {'Enabled' if config.USE_GPU else 'Disabled'}")
    print(f"🎯 Detection Side: {config.DETECTION_SIDE.upper()}")
    initialize_model()
    initialize_video_source()
    print(f"✅ System ready!")


@app.on_event("shutdown")
async def shutdown_event():
    global cap
    if cap is not None:
        cap.release()
    print("👋 System shutdown")


@app.get("/")
async def root():
    return {
        "message": "Person Tracking System with Side Detection",
        "version": "7.0",
        "features": ["Angled Boundary (0-360°)", "Side Detection (Left/Right/Both)", "Dynamic Control"]
    }


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/latest_crossing")
async def get_latest_crossing():
    if latest_crossing:
        return latest_crossing
    return {"message": "No crossings detected yet"}


@app.get("/stats")
async def get_stats():
    return {
        "total_tracked": len(tracker.tracked_persons),
        "total_alerts": len(tracker.crossed_ids),
        "boundary_angle": config.BOUNDARY_ANGLE,
        "boundary_ratio": config.BOUNDARY_LINE_RATIO,
        "detection_side": config.DETECTION_SIDE
    }


@app.post("/config/boundary")
async def update_boundary(
        ratio: Optional[float] = None,
        angle: Optional[float] = None,
        side: Optional[str] = None
):
    """Update boundary configuration including detection side"""
    if ratio is not None:
        if not 0.0 <= ratio <= 1.0:
            raise HTTPException(status_code=400, detail="Ratio must be between 0.0 and 1.0")
        config.BOUNDARY_LINE_RATIO = ratio

    if angle is not None:
        config.BOUNDARY_ANGLE = angle % 360

    if side is not None:
        side_lower = side.lower()
        if side_lower not in ["left", "right", "both"]:
            raise HTTPException(status_code=400, detail="Side must be 'left', 'right', or 'both'")
        config.DETECTION_SIDE = side_lower
        print(f"🎯 Detection side updated to: {side_lower.upper()}")

    return {
        "message": "Configuration updated",
        "ratio": config.BOUNDARY_LINE_RATIO,
        "angle": config.BOUNDARY_ANGLE,
        "detection_side": config.DETECTION_SIDE
    }


@app.post("/reset")
async def reset_tracking():
    global tracker
    tracker = PersonTracker()
    return {
        "message": "Tracking reset",
        "tracked": 0,
        "alerts": 0,
        "detection_side": config.DETECTION_SIDE
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)