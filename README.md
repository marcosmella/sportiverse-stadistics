# sportiverse-stadistics

# 🎾 Pádel Match Analyzer MVP — Detection + Tracking + Supabase

This project detects and tracks pádel players in match videos using computer vision, and stores basic statistics in a Supabase database. It’s a minimal viable product (MVP) using **YOLOv8** for detection, **DeepSORT** for tracking, and **Supabase** as a backend.

---

## ✅ MVP Goals

1. 📹 Load a pádel match video  
2. 🧠 Detect players in real-time using YOLOv8  
3. 🧾 Track players using DeepSORT (for continuity)  
4. 📊 Calculate basic movement stats (distance, zones)  
5. ☁️ Send this data to Supabase for use in your web app

---

## 🧠 Why These Tools?

| Component | Tool | Reason |
|----------|------|--------|
| Detection | YOLOv8 | Fast, real-time person detection |
| Tracking | DeepSORT | Maintains consistent IDs across frames |
| Backend | Supabase | Scalable, simple PostgreSQL + REST API |
| Processing | OpenCV | Lightweight video reading, frame drawing |

---

## 📦 1. Installation

### 1.1 Clone this repository (or create your own)
```bash
git clone https://github.com/your-username/padel-analyzer-mvp.git
cd padel-analyzer-mvp
```
1.2 Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```
1.3 Install dependencies
```bash
pip install -r requirements.txt
```
requirements.txt:
ultralytics
opencv-python
deep_sort_realtime
supabase
python-dotenv

 Why YOLOv8?
YOLOv8 supports PyTorch and is optimized for real-time detection.

It includes a pre-trained model that recognizes “person” and “sports ball” from the COCO dataset.

❓ Why DeepSORT?

YOLO detects each frame independently.

DeepSORT adds tracking: it gives each player a unique ID and follows them across time, allowing us to compute per-player stats.

 2. Add Your Pádel Video
Place your match video file in the project folder and name it:
padel_match.mp4 (VIDEO)

If you use a different name, update it in the script detectar_y_trackear.py.

🧠 3. Run Detection + Tracking

python detectar_y_trackear.py

This script does the following:

Loads padel_match.mp4

Uses YOLOv8 to detect people

Uses DeepSORT to track each player over time

Draws bounding boxes and IDs on video

Optionally collects stats per player (distance, heatmap zone, etc.)

📍 File: detectar_y_trackear.py (simplified logic)
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

model = YOLO("yolov8n.pt")  # Lightweight model for MVP
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("padel_match.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'Jugador {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Tracking Jugadores", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()


☁️ 4. Supabase Integration (Export Stats)
Copy the URL and anon/public key

4.4 Supabase schema
create table player_stats (
  id serial primary key,
  match_id integer,
  player_id integer,
  track_id integer,
  distance_moved_m float,
  zone_time jsonb,
  created_at timestamp with time zone default now()
);

Table: player_stats

Field	Type	Description
id	int (PK)	Auto-increment ID
match_id	int	ID of the match
player_id	int	Arbitrary or internal ID
track_id	int	Track ID from DeepSORT
distance_moved_m	float	Total movement (approx)
zone_time	JSONB	Time spent in zones
created_at	timestamptz	Timestamp

You can manage this schema in Supabase UI or via SQL script.

4.5 Export example: send_stats_to_supabase.py

from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

data = {
    "match_id": 1,
    "player_id": 2,
    "track_id": 7,
    "distance_moved_m": 125.6,
    "zone_time": {
        "left_back": 32.4,
        "right_back": 25.0,
        "net": 12.1
    }
}

response = supabase.table("player_stats").insert(data).execute()
print(response)

🚀 Next Steps
Feature	Description
🔍 Ball tracking	Use YOLO to track ball trajectory (YOLOv8 can detect sports ball)
🧩 Highlight extraction	Detect point rallies (ball crosses net, ends in bounce)
📊 Heatmaps	Use OpenCV + numpy to create movement density maps per player
🎥 Live streaming	Integrate with RTSP/IP cameras or OBS
🧱 Web dashboard	Build React or Next.js app to show stats and video highlights

📁 Project Structure

padel-analyzer-mvp/
├── padel_match.mp4                # Your input video
├── detectar_y_trackear.py        # Detection & tracking script
├── send_stats_to_supabase.py     # Exports stats to Supabase
├── .env                          # Your Supabase keys
├── requirements.txt
├── README.md





