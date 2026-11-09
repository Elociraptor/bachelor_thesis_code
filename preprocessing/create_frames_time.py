'''
create_frames.py
Dieses Modul erstellt die Keyframes für die feste Frame-Strategie aus den Videos.

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''
import cv2
import os
from pathlib import Path
from rich.progress import track

def extract_frames(video_path, output_dir, fps, video_name="video"):

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Konnte Video nicht öffnen: {video_path}")
        return
    vid_fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(vid_fps / fps)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, frame_count, frame_interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"{video_name}_frame_{i:05d}.jpg"), frame)

    video.release()


def extract_all_frames_folder(folder_path, output_dir, fps):
    for filename in track(os.listdir(folder_path)):
        if filename.endswith(".mp4"):
            video_path = os.path.join(folder_path, filename)
            video_name = os.path.splitext(filename)[0]
            extract_frames(video_path, output_dir, fps, video_name)

BASE_DIR = Path(__file__).parent.parent
extract_all_frames_folder(BASE_DIR / "data/charades/videos", BASE_DIR / "data/charades_frames_1fps", 1)