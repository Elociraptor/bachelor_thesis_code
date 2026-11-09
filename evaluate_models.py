'''
evaluate_models.py
Dieses Modul führt die automatische Evaluation für jede Anfrage aus und bestimmt die Metriken.
Die Ergebnisse werden an die Ergebnis CSV angehängt.

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''
import json
import csv
import random
import torch
import faiss
import numpy as np
import open_clip
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

import config
import sys
ANNOTATIONS = config.ANNOTATIONS
RESULTS_CSV = config.RESULTS_CSV
DEDUP_PER_VIDEO = True
QUERY_BATCH = 256
SEED = 42

def load_captions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    captions = {}
    for entry in data["annotations"]:
        captions.setdefault(entry["image_id"], []).append(entry["caption"])
    return captions

def main():
    print(f"Device: {config.DEVICE}\n")
    random.seed(SEED)

    model, _ = config.load_model()
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

    index = faiss.read_index(str(config.INDEX_FILE))
    with open(config.METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"{index.ntotal} Frames im Index")

    indexed_videos = sorted({m["video_id"] for m in metadata})
    print(f"{len(indexed_videos)} Videos im Index\n")

    captions = load_captions(ANNOTATIONS)
    query_videos = []
    query_texts  = []
    for vid_id in indexed_videos:
        if vid_id in captions and captions[vid_id]:
            query_videos.append(vid_id)
            query_texts.append(random.choice(captions[vid_id]))

    print(f"{len(query_texts)} Anfragen werden Evaluiert")

    # Die Querie Embeddings bestimmen
    query_embeddings = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Encodiere Text Queries", total=len(query_texts))
        for start in range(0, len(query_texts), QUERY_BATCH):
            batch = query_texts[start : start + QUERY_BATCH]
            tokens = tokenizer(batch).to(config.DEVICE)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
            query_embeddings.append(feats.cpu().numpy())
            progress.update(task, advance=len(batch))
    query_np = np.concatenate(query_embeddings, axis=0).astype("float32")

    #Das richtige Zielvideo finden
    ranks = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Suche und Ranke Anfragen", total=len(query_videos))
        for i, target_video in enumerate(query_videos):
            scores, indices = index.search(query_np[i:i+1], index.ntotal)
            indices = indices[0]
            rank = None
            seen = set()
            position = 0
            for idx in indices:
                if idx == -1:
                    continue
                vid = metadata[idx]["video_id"]
                if DEDUP_PER_VIDEO and vid in seen:
                    continue
                seen.add(vid)
                position += 1
                if vid == target_video:
                    rank = position
                    break

            if rank is not None:
                ranks.append(rank)

            progress.update(task, advance=1)

    ranks = np.array(ranks)

    #Metriken
    r1 = np.mean(ranks <= 1)
    r5 = np.mean(ranks <= 5)
    r10 = np.mean(ranks <= 10)
    median_rank = np.median(ranks)
    mean_rank = np.mean(ranks)

    print(f"\nErgebnisse für {config.MODEL_TAG}")
    print(f"Anzahl der Anfragen: {len(ranks)}")
    print(f"R@1: {r1:.4f}")
    print(f"R@5: {r5:.4f}")
    print(f"R@10: {r10:.4f}")
    print(f"Median Rank: {median_rank:.1f}")
    print(f"Mean Rank: {mean_rank:.2f}")

    #Ergebnis an CSV anhängen:
    #Im folgenden Code abschnitt wurde Claude Opus 4.8 zur Unterstützung eingesetzt
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model_tag", "num_queries", "R@1", "R@5", "R@10",
                             "median_rank", "mean_rank"])
        writer.writerow([config.MODEL_TAG, len(ranks),
                         f"{r1:.4f}", f"{r5:.4f}", f"{r10:.4f}",
                         f"{median_rank:.1f}", f"{mean_rank:.2f}"])
    print(f"\n-> an {RESULTS_CSV} angehaengt")


if __name__ == "__main__":
    main()