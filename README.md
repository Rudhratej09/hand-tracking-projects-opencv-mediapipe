# Hand Tracking Projects with OpenCV & MediaPipe

This repository contains multiple computer vision mini-projects built using [MediaPipe](https://google.github.io/mediapipe/) and [OpenCV](https://opencv.org/) focused on **real-time hand detection, gesture recognition**, and interactive applications.

## 📁 Projects Included

| Project | Description |
|--------|-------------|
| `finger_counter/` | Detects how many fingers are held up using hand landmarks. |
| `gesture_volume_control/` | Control your system volume using distance between thumb and index finger. |
| `hand_tracking_module.py` | Modular hand detection logic reused across all projects. |

---

## Tech Stack
- **Python 3**
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [pycaw](https://github.com/AndreMiras/pycaw) *(for audio control)*

---

## Demo Screenshots

<p align="center">
  <img width="1917" height="1118" alt="image" src="https://github.com/user-attachments/assets/497361ac-5788-45d4-9c85-9ce4108e8e1b" />

  <<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/9c2c1cd5-78e1-43fc-95b8-c4f5179b860c" />
>
</p>

---

## 🛠️ Setup Instructions

1. Clone the repo:
   
  git clone https://github.com/yourusername/hand-tracking-projects-opencv-mediapipe.git
  
  cd hand-tracking-projects-opencv-mediapipe

2.install dependencies

  pip install -r requirements.txt

3.run a project 

  cd finger_counter
  
  python finger_counter.py

