'''
create_frames_transnetv2.py
Dieses Modul ermittelt die Szenengrenzen mit TransNetV2 und
extrahiert die Keyframes aus den Videos am Szenenmittelpunkt.

Dieser Code wurde mithilfe des TransNetV2 Repos geschrieben. (https://github.com/soCzech/TransNetV2)

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''
import torch
import cv2
import numpy as np
from transnetv2_pytorch import TransNetV2
from pathlib import Path
from rich.progress import track

BASE_DIR = Path(__file__).parent.parent
VIDEO_DIR = BASE_DIR / "data/videos"
OUTPUT_DIR = BASE_DIR / "data/video_frames_transnetv2"
OUTPUT_DIR.mkdir(exist_ok=True)

model = TransNetV2()
state_dict = torch.load(BASE_DIR / "models/transnetv2-pytorch-weights.pth")
model.load_state_dict(state_dict)
model.eval().cuda()

def load_frames_smaller(video_path) -> np.ndarray:
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Konnte Video nicht öffnen: {video_path}")
        return
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame_rgb, (48, 27)))
    video.release()
    return np.array(frames)

def predictions_to_scenes(predictions, threshold=0.5, min_scene_length=5):
    cut_frames = np.where(predictions > threshold)[0]
    scenes = []
    start = 0
    for cut in cut_frames:
        if (cut - start) >= min_scene_length:
            scenes.append((start, cut))
            start = cut + 1
    scenes.append((start, len(predictions)-1))
    return scenes

def extract_original_frame(video_path, frame_number):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Konnte Video nicht öffnen: {video_path}")
        return
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    video.release()
    if not ret:
        raise ValueError(f"Frame {frame_number} konnte in Video: {video_path} nicht gefunden werden.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

video_extention = ".mp4"
video_files = list(VIDEO_DIR.glob(f"**/*{video_extention}"))

for video_path in track(video_files):
    frames_smaller = load_frames_smaller(video_path)
    input_video = torch.from_numpy(frames_smaller).unsqueeze(0).cuda()
    with torch.no_grad():
        single_frame_pred, all_frame_pred = model(input_video)
    predictions = torch.sigmoid(single_frame_pred).squeeze().cpu().numpy()
    scenes = predictions_to_scenes(predictions)
    for start, end in scenes:
        midpoint = (start + end) // 2
        original_frame = extract_original_frame(video_path, midpoint)
        cv2.imwrite(OUTPUT_DIR / f"{video_path.stem}_{start:05d}.jpg", cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))
