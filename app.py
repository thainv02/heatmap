import cv2
import numpy as np
import json
import os
import uuid
from ultralytics import YOLO
from datetime import datetime
import time
import signal
import sys

RTSP_URL = "rtsp://test:zholding2024@hgw0aekf8e3.sn.mynetname.net:1554/Streaming/Channels/501"
FRAME_SKIP = 5
JSON_SAVE_INTERVAL = 300
MAX_TRACKING_AGE = 30

ROI = np.array([[590, 11], [16, 651], [104, 1065], [1910, 1072], [1910, 294], [1152, 0]], np.int32)
ROI = ROI.reshape((-1, 1, 2))

TARGET_CLASS = 0

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"heatmap_data_trandangninh"
os.makedirs(output_dir, exist_ok=True)

log_file = os.path.join(output_dir, f"log_{session_id}.txt")
def log_message(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

log_message(f"Starting RTSP data collection: {session_id}")
log_message(f"RTSP URL: {RTSP_URL}")
log_message(f"Output directory: {output_dir}")
log_message(f"Target class: {TARGET_CLASS}")
log_message(f"ROI configured with {len(ROI)} points")

cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    log_message("Error: Cannot connect to RTSP stream. Exiting.")
    sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

log_message(f"Stream properties: {w}x{h} at {fps} FPS")

stream_metadata = {
    "session_id": session_id,
    "rtsp_url": RTSP_URL,
    "width": w,
    "height": h,
    "fps": fps,
    "start_time": datetime.now().isoformat(),
    "roi": ROI.reshape(-1, 2).tolist(),
    "target_class": TARGET_CLASS
}

metadata_file = os.path.join(output_dir, "metadata_thaithinh.json")
with open(metadata_file, 'w') as f:
    json.dump(stream_metadata, f, indent=2)
log_message(f"Saved stream metadata to {metadata_file}")

log_message("Initializing YOLO model")
detector = YOLO("best_yolo11x_KH.pt")

class ObjectTracker:
    def __init__(self):
        self.objects = {}
        self.last_seen = {}
        self.completed_objects = []
    
    def update(self, current_frame, detections):
        self._check_expired_objects(current_frame)
        for det in detections:
            center_x, center_y = det["center_x"], det["center_y"]
            matched = False
            min_dist = float('inf')
            closest_id = None
            for track_id, obj in self.objects.items():
                last_pos = obj["timestamps"][-1]
                last_x = last_pos.get("center_x", 0)
                last_y = last_pos.get("center_y", 0)
                dist = ((center_x - last_x) ** 2 + (center_y - last_y) ** 2) ** 0.5
                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    closest_id = track_id
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            if closest_id is not None:
                self.objects[closest_id]["timestamps"].append({
                    "time": timestamp,
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"]
                })
                self.last_seen[closest_id] = current_frame
            else:
                new_id = str(uuid.uuid4())
                self.objects[new_id] = {
                    "tracking_id": new_id,
                    "timestamps": [{
                        "time": timestamp,
                        "center_x": center_x,
                        "center_y": center_y,
                        "bbox": det["bbox"],
                        "confidence": det["confidence"]
                    }],
                    "entry_time": timestamp,
                    "exit_time": None
                }
                self.last_seen[new_id] = current_frame
    
    def _check_expired_objects(self, current_frame):
        expired_ids = []
        for track_id, last_frame in self.last_seen.items():
            if current_frame - last_frame > MAX_TRACKING_AGE:
                expired_ids.append(track_id)
        for track_id in expired_ids:
            if track_id in self.objects:
                obj = self.objects[track_id]
                obj["exit_time"] = obj["timestamps"][-1]["time"]
                self.completed_objects.append(obj)
                del self.objects[track_id]
                del self.last_seen[track_id]
    
    def save_to_json(self, filepath):
        all_objects = self.completed_objects.copy()
        for track_id, obj in self.objects.items():
            obj_copy = obj.copy()
            obj_copy["exit_time"] = obj["timestamps"][-1]["time"] if obj["timestamps"] else None
            all_objects.append(obj_copy)
        with open(filepath, 'w') as f:
            json.dump(all_objects, f, indent=2)
        return len(all_objects)

tracker = ObjectTracker()
main_json_file = os.path.join(output_dir, "tracking_data.json")

def is_point_in_roi(point, roi):
    return cv2.pointPolygonTest(roi, point, False) >= 0

def signal_handler(sig, frame):
    log_message(f"Received signal {sig}. Cleaning up and exiting...")
    num_objects = tracker.save_to_json(main_json_file)
    log_message(f"Saved tracking data for {num_objects} objects to {main_json_file}")
    end_metadata = {
        "end_time": datetime.now().isoformat(),
        "frames_processed": frames_processed,
        "total_runtime_seconds": time.time() - start_time
    }
    end_file = os.path.join(output_dir, "end_metadata.json")
    with open(end_file, 'w') as f:
        json.dump(end_metadata, f, indent=2)
    cap.release()
    log_message(f"Session completed. Processed {frames_processed} frames.")
    log_message(f"All data saved to directory: {output_dir}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

frame_count = 0
skip_count = 0
start_time = time.time()
frames_processed = 0
last_json_save = 0

last_reconnect_time = 0
reconnect_cooldown = 5

try:
    log_message("Starting main processing loop")
    while True:
        success, im0 = cap.read()
        if not success:
            current_time = time.time()
            if current_time - last_reconnect_time > reconnect_cooldown:
                log_message("Failed to read from RTSP stream. Attempting to reconnect...")
                cap.release()
                time.sleep(reconnect_cooldown)
                cap = cv2.VideoCapture(RTSP_URL)
                last_reconnect_time = time.time()
                if not cap.isOpened():
                    log_message("Reconnection failed. Will retry...")
            continue
        skip_count += 1
        if skip_count % FRAME_SKIP != 0:
            continue
        try:
            results = detector(im0, conf=0.5)
            frame_objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    if class_id != TARGET_CLASS:
                        continue
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    is_in_roi = is_point_in_roi((center_x, center_y), ROI)
                    if not is_in_roi:
                        continue
                    obj_data = {
                        "class_id": class_id,
                        "confidence": confidence,
                        "center_x": center_x,
                        "center_y": center_y,
                        "bbox": [x1, y1, x2, y2],
                        "in_roi": is_in_roi
                    }
                    frame_objects.append(obj_data)
            tracker.update(frame_count, frame_objects)
            frame_count += 1
            frames_processed += 1
            if frame_count - last_json_save >= JSON_SAVE_INTERVAL:
                num_objects = tracker.save_to_json(main_json_file)
                log_message(f"Saved tracking data for {num_objects} objects to {main_json_file}")
                last_json_save = frame_count
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps_processing = frames_processed / elapsed_time if elapsed_time > 0 else 0
                memory_usage = 0
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_usage = process.memory_info().rss / (1024 * 1024)
                except ImportError:
                    pass
                log_message(f"Processed {frames_processed} frames in {elapsed_time:.2f}s ({fps_processing:.2f} FPS), Memory: {memory_usage:.2f} MB")
        except Exception as e:
            log_message(f"Error processing frame {frame_count}: {str(e)}")
            continue
except Exception as e:
    log_message(f"Unhandled exception: {str(e)}")
finally:
    signal_handler(signal.SIGTERM, None)
