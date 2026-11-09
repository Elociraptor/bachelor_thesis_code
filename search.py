'''
search.py
Dieses Modul wurde zum anfänglichen Testen des Systems entwickelt.
Es nimmt Textanfragen ber CLI-Input entgegen und gibt die besten K Videos aus.
Dieses Modul ist nicht Teil der endgültigen Pipeline.

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''
import json
import torch
import faiss
import open_clip
from pathlib import Path
import config

#Such-Einstellungen
TOP_K_VIDEOS = 5
FETCH_FRAMES = 100

# True  = pro Video nur bester Frame
# False = dasselbe Video darf mehrfach erscheinen
DEDUP_PER_VIDEO = True


def search(query_text, model, tokenizer, index, metadata):
    tokens = tokenizer([query_text]).to(config.DEVICE)
    with torch.no_grad():
        query_features = model.encode_text(tokens)
        query_features /= query_features.norm(dim=-1, keepdim=True)
    query_np = query_features.cpu().numpy().astype("float32")

    scores, indices = index.search(query_np, FETCH_FRAMES)
    scores, indices = scores[0], indices[0]

    seen_videos = set()
    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        info = metadata[idx]
        vid_id = info["video_id"]

        if DEDUP_PER_VIDEO:
            if vid_id in seen_videos:
                continue
            seen_videos.add(vid_id)

        results.append({
            "video_id": vid_id,
            "frame_path": info["frame_path"],
            "score": float(score),
        })
        if len(results) >= TOP_K_VIDEOS:
            break

    return results


def main():
    print(f"Device: {config.DEVICE}\n")

    model, _ = config.load_model()
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

    print("Lade Index und Metadaten...")
    index = faiss.read_index(str(config.INDEX_FILE))
    with open(config.METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"{index.ntotal} Frames im Index\n")

    print("Suchtext eingeben:\n")
    while True:
        query = input("Suche> ").strip()
        if query.lower() in ("q", "quit"):
            break

        results = search(query, model, tokenizer, index, metadata)
        print(f"\nTop {len(results)} Ergebnisse fuer: '{query}'")
        for rank, r in enumerate(results, start=1):
            frame_name = Path(r["frame_path"]).name
            print(f"  {rank}. {r['video_id']:12s}  Score: {r['score']:.4f}  ({frame_name})")
        print()


if __name__ == "__main__":
    main()