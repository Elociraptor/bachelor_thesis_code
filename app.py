'''
app.py
Dieses Modul stellt die App für das interaktive Video-Retrieval dar.
In der App kann per Button zwischen zwei Modi gewechselt werden: der Einzelsuche
und dem Vergleich von zwei Konfigurationen.
Bei den Konfigurationen kann jewils eine Frame-Strategie, ein alpha-Wert und die Frame-Strategie
gewählt werden.

Gestartet wird die App mit: streamlit run app.py

Bei der Erstellung des App-Codes wurde Claude Opus 4.8 zur Unterstützung eingesetzt.

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''

import time
import json
import torch
import faiss
import open_clip
import streamlit as st
from pathlib import Path
from PIL import Image

import config

MODELS = {
    "datacomp-L, Partial (Ueki)": {
        "dir": "wiseft_ueki_ep7",
        "tag": "wiseft_ueki_ep7",
        "pretrained": "datacomp_l_s1b_b8k",
    },
    "datacomp-L, Full FT": {
        "dir": "wiseft_full_ft_ep7",
        "tag": "wiseft_full_ft_ep7",
        "pretrained": "datacomp_l_s1b_b8k",
    },
    "datacomp-XL, Partial (Ueki)": {
        "dir": "wiseft_xl_ueki_ep7",
        "tag": "wiseft_xl_ueki_ep7",
        "pretrained": "datacomp_xl_s13b_b90k",
    },
    "datacomp-XL, Full FT": {
        "dir": "wiseft_xl_full_ft_ep6",
        "tag": "wiseft_xl_full_ft_ep6",
        "pretrained": "datacomp_xl_s13b_b90k",
    },
}

STRATEGIES = {
    "2 fps":      "2fps",
    "1 fps":      "1s",
    "0.5 fps":    "05s",
    "TransNetV2": "transnetv2",
}
ALPHAS = [round(0.1 * i, 1) for i in range(11)]

def index_dir_for(model_key, alpha, strat_key):
    tag = f"{MODELS[model_key]['tag']}_alpha{alpha}_{strat_key}"
    return config.DATA_DIR / f"index_{tag}"

def checkpoint_for(model_key, alpha):
    return Path("logs") / MODELS[model_key]["dir"] / f"alpha_{alpha}.pt"

@st.cache_resource(show_spinner="Lade Modell...")
def load_model_cached(checkpoint_path, pretrained):
    model, _, _ = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=pretrained
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(config.DEVICE).eval()

@st.cache_resource(show_spinner="Lade Index...")
def load_index_cached(index_dir_str):
    index_dir = Path(index_dir_str)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

@st.cache_resource
def get_tokenizer():
    return open_clip.get_tokenizer(config.MODEL_NAME)

def search(query, model, tokenizer, index, metadata, top_k, fetch, dedup):
    #Encoding-Zeit
    t0 = time.perf_counter()
    tokens = tokenizer([query]).to(config.DEVICE)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    query_np = feats.cpu().numpy().astype("float32")
    encode_ms = (time.perf_counter() - t0) * 1000

    #Such-Zeit
    t0 = time.perf_counter()
    scores, indices = index.search(query_np, fetch)
    search_ms = (time.perf_counter() - t0) * 1000

    scores, indices = scores[0], indices[0]

    seen = set()
    results = []
    for score, idx in zip(scores, indices):
        if idx == -1:
            continue
        info = metadata[idx]
        vid = info["video_id"]
        if dedup and vid in seen:
            continue
        seen.add(vid)
        results.append({"video_id": vid, "frame_path": info["frame_path"],
                        "score": float(score)})
        if len(results) >= top_k:
            break
    return results, encode_ms, search_ms

def render_config(prefix, default_alpha=0.7):
    model_key = st.selectbox("Modell", list(MODELS.keys()), key=f"{prefix}_model")
    alpha = st.select_slider("alpha", options=ALPHAS, value=default_alpha,
                             key=f"{prefix}_alpha",
                             help="0.0 = Zero-Shot, 1.0 = reines Fine-Tuning")
    strat_label = st.selectbox("Frame-Strategie", list(STRATEGIES.keys()),
                               key=f"{prefix}_strat")
    return model_key, alpha, STRATEGIES[strat_label]

def run_and_show(query, model_key, alpha, strat_key, top_k, dedup, cols_per_row=4):
    index_dir  = index_dir_for(model_key, alpha, strat_key)
    checkpoint = checkpoint_for(model_key, alpha)

    if not (index_dir / "index.faiss").exists():
        st.error(f"Kein Index für diese Kombination:\n`{index_dir.name}`")
        return
    if not checkpoint.exists():
        st.error(f"Kein Checkpoint:\n`{checkpoint}`")
        return

    model = load_model_cached(str(checkpoint), MODELS[model_key]["pretrained"])
    index, metadata = load_index_cached(str(index_dir))
    tokenizer = get_tokenizer()

    if query:
        results, encode_ms, search_ms = search(
            query, model, tokenizer, index, metadata,
            top_k=top_k, fetch=200, dedup=dedup,
        )

        c1, c2 = st.columns(2)
        c1.metric("Encoding", f"{encode_ms:.1f} ms")
        c2.metric("Suche", f"{search_ms:.2f} ms")

        for row_start in range(0, len(results), cols_per_row):
            row = results[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, r in zip(cols, row):
                with col:
                    fp = Path(r["frame_path"])
                    if fp.exists():
                        st.image(str(fp), use_container_width=True)
                    else:
                        st.write("(Bild fehlt)")
                    st.caption(f"{r['video_id']} · {r['score']:.3f}")
    else:
        st.info(f"Index: {index.ntotal:,} Frames · "
                f"{len({m['video_id'] for m in metadata}):,} Videos")


#Oberfläche
st.set_page_config(page_title="Video-Retrieval", layout="wide")
st.title("CLIP-basiertes Text-zu-Video-Retrieval")

with st.sidebar:
    mode = st.radio("Modus", ["Einzelsuche", "Vergleich"])
    st.divider()
    st.header("Such-Optionen")
    top_k = st.slider("Anzahl Ergebnisse", 1, 20, 8)
    dedup = st.checkbox("Pro Video nur bestes Bild", value=True)

#Query
query = st.text_input("Suchanfrage",
                      placeholder="a professor grading his students thesis")

if mode == "Einzelsuche":
    with st.sidebar:
        st.divider()
        st.header("Modell")
        fam, alpha, strat = render_config("single")
    run_and_show(query, fam, alpha, strat, top_k, dedup, cols_per_row=4)

else:
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Modell A")
        famA, alphaA, stratA = render_config("A", default_alpha=0.0)
        st.divider()
        run_and_show(query, famA, alphaA, stratA, top_k, dedup, cols_per_row=2)
    with colB:
        st.subheader("Modell B")
        famB, alphaB, stratB = render_config("B", default_alpha=0.7)
        st.divider()
        run_and_show(query, famB, alphaB, stratB, top_k, dedup, cols_per_row=2)