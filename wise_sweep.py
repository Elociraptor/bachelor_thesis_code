'''
wise_sweep.py
Dieses Modul führt den automatischen Sweep über alle alphas und Frame-Strategien durch und ruft dazu die
Module build_image_vectors.py, build_faiss_index.py und evaluate_models.py auf.
Das Modul make_wiseft.py muss vorher gelaufen sein.

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''

import csv
from pathlib import Path
import config
import build_image_vectors
import build_faiss_index
import evaluate_models

FT_TAG     = "full_ft_ep6"
WISEFT_DIR = Path("logs") / f"wiseft_xl_{FT_TAG}"

ALPHAS = [round(0.1 * i, 1) for i in range(11)]

FRAME_STRATEGIES = {
    "charades_1s": config.DATA_DIR / "charades_frames_1fps",
#    "transnetv2": config.DATA_DIR / "video_frames_transnetv2",
#    "05s": config.DATA_DIR / "video_frames_05s",
#    "2fps": config.DATA_DIR / "video_frames_2fps",
}

RESULTS_CSV = config.DATA_DIR / "eval_results_charades.csv"
TEST_SPLIT = config.DATA_DIR / "splits" / "test_list_charades.txt"


def already_evaluated(model_tag):
    if not RESULTS_CSV.exists():
        return False
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] == model_tag:
                return True
    return False

def main():
    total = len(ALPHAS) * len(FRAME_STRATEGIES)
    done = 0

    for strat_name, frames_dir in FRAME_STRATEGIES.items():
        for alpha in ALPHAS:
            done += 1
            tag = f"charades_wiseft_{FT_TAG}_alpha{alpha}_{strat_name}"
            source = WISEFT_DIR / f"alpha_{alpha}.pt"

            config.configure(model_source=str(source), model_tag=tag, frames_dir=frames_dir, test_split=TEST_SPLIT )

            if config.INDEX_FILE.exists():
                print("Index existiert bereits")
            else:
                build_image_vectors.main()
                build_faiss_index.main()

            if already_evaluated(tag):
                print("Wurde bereits evaluiert")
            else:
                evaluate_models.main()

    print(f"Sweep fertig. Ergebnisse in {RESULTS_CSV} gespeichert.")


if __name__ == "__main__":
    main()