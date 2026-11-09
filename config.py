'''
config.py
Dieses Modul ist die zentrale Konfigurationsdatei.

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''
import torch
import open_clip
from pathlib import Path

MODEL_NAME = "ViT-B-16"
PRETRAINED = "datacomp_xl_s13b_b90k"
#Verwendetes Model:
MODEL_SOURCE = "logs/wiseft_xl_ueki_ep7/alpha_0.5.pt"
#MODEL_SOURCE = "zeroshot"

MODEL_TAG = "wiseft_ueki_ep7_alpha0.5_transnetv2"

DATA_DIR = Path("data")
TEST_SPLIT = DATA_DIR / "splits" / "test_list_full.txt"
#FRAMES_DIR = DATA_DIR / "video_frames_1s"
FRAMES_DIR = DATA_DIR / "video_frames_transnetv2"

ANNOTATIONS = DATA_DIR / "annotations_charades.json"
RESULTS_CSV = DATA_DIR / "eval_results_charades.csv"

INDEX_DIR = EMBEDDINGS_FILE = METADATA_FILE = INDEX_FILE = None

LIMIT_VIDEOS = None
BATCH_SIZE   = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _refresh():
    global INDEX_DIR, EMBEDDINGS_FILE, METADATA_FILE, INDEX_FILE
    INDEX_DIR = DATA_DIR / f"index_{MODEL_TAG}"
    EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
    METADATA_FILE = INDEX_DIR / "metadata.json"
    INDEX_FILE = INDEX_DIR / "index.faiss"

def configure(model_source=None, model_tag=None, frames_dir=None, test_split=None):
    global MODEL_SOURCE, MODEL_TAG, FRAMES_DIR, TEST_SPLIT
    if model_source is not None:
        MODEL_SOURCE = model_source
    if model_tag is not None:
        MODEL_TAG = model_tag
    if frames_dir is not None:
        FRAMES_DIR = frames_dir
    if test_split is not None:
        TEST_SPLIT = test_split
    _refresh()

def load_model():
    if MODEL_SOURCE == "zeroshot":
        print("Lade Zero-Shot Modell")
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
    else:
        print(f"Lade Checkpoint: {MODEL_SOURCE}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        checkpoint = torch.load(MODEL_SOURCE, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

    model = model.to(DEVICE).eval()
    return model, preprocess

_refresh()