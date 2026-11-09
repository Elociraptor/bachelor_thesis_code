'''
implement_wiseft.py
Dieses Modul erstellt alle interpolierten Modelle nach der WiSE-FT Methode.
FÜr jedes alpha wird ein Checkpoint erzeugt.

Dieser Code ist am originalen Code des WiSE-FT-Repos inspiriert. (https://github.com/mlfoundations/wise-ft)

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''
import torch
import open_clip
from pathlib import Path

import config

FINETUNED_CHECKPOINT = "logs/full_ft_10ep/checkpoints/epoch_7.pt"
FT_TAG = "full_ft_ep7"
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
OUTPUT_DIR = Path("logs") / f"wiseft_{FT_TAG}"


def load_state_dict_from_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Lade Zero-Shot Modell")
    model_0, _, _ = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.PRETRAINED
    )
    theta_0 = model_0.state_dict()

    print(f"Lade feingetuntes Modell: {FINETUNED_CHECKPOINT}")
    theta_1 = load_state_dict_from_checkpoint(FINETUNED_CHECKPOINT)

    assert set(theta_0.keys()) == set(theta_1.keys()), \
        "Keys von theta_0 und theta_1 unterschiedlich!"

    for alpha in ALPHAS:
        theta_interp = {}
        for key in theta_0.keys():
            t0 = theta_0[key]
            t1 = theta_1[key]

            if t0.is_floating_point():
                theta_interp[key] = (1 - alpha) * t0 + alpha * t1
            else:
                theta_interp[key] = t1

        out_path = OUTPUT_DIR / f"alpha_{alpha}.pt"
        torch.save({"state_dict": theta_interp}, out_path)

    print(f"{len(ALPHAS)} Checkpoints in {OUTPUT_DIR} gespeichert")


if __name__ == "__main__":
    main()