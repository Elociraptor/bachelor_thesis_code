'''
build_image_vectors.py
Dieses Modul lädt alle Frames und erzeugt ihre Embeddings.
Die Embeddings werden in ein numpy-Array gespeichert.

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''
import json
import torch
import numpy as np
from PIL import Image
import config

def load_ids(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    print(f"Device: {config.DEVICE}\n")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocess = config.load_model()
    print("  geladen\n")

    test_ids = load_ids(config.TEST_SPLIT)
    if config.LIMIT_VIDEOS is not None:
        test_ids = test_ids[:config.LIMIT_VIDEOS]
    print(f"Verarbeite {len(test_ids)} Videos")

    frame_paths = []
    metadata = []
    videos_with_frames = 0
    videos_without_frames = []
    for vid_id in test_ids:
        frames = sorted(config.FRAMES_DIR.glob(f"{vid_id}_*.jpg"))
        if frames:
            videos_with_frames += 1
        else:
            videos_without_frames.append(vid_id)
        for frame_path in frames:
            frame_paths.append(frame_path)
            metadata.append({
                "video_id": vid_id,
                "frame_path": str(frame_path).replace("\\", "/"),
            })

    if videos_without_frames:
        print(f"Warnung: {len(videos_without_frames)} Videos ohne Frames")

    print(f"insgesamt {len(frame_paths)} Frames geladen\n")

    all_embeddings = []
    print("Erzeuge Embeddings")
    for start in range(0, len(frame_paths), config.BATCH_SIZE):
        batch_paths = frame_paths[start : start + config.BATCH_SIZE]

        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        image_batch = torch.stack(images).to(config.DEVICE)

        with torch.no_grad():
            features = model.encode_image(image_batch)
            features /= features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())
        print(f"  {min(start + config.BATCH_SIZE, len(frame_paths))}/{len(frame_paths)} Frames")

    embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")

    np.save(config.EMBEDDINGS_FILE, embeddings)
    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Fertig.")
    print(f"Modell:{config.MODEL_TAG}")
    print(f"Gespeichert in:{config.INDEX_DIR}/")


if __name__ == "__main__":
    main()