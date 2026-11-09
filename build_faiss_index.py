'''
build_faiss_index.py
Dieses Modul lädt die Embeddings, erstellt einen Index und speichert ihn.

Teil der Bachelorarbeit: "Optimierungsstrategien für CLIP-basierte Text-zu-Video-Retrieval-Systeme"
von Ole Marwede
'''
import faiss
import numpy as np
import config


def main():
    print(f"Lade Embeddings aus {config.EMBEDDINGS_FILE}...")
    embeddings = np.load(config.EMBEDDINGS_FILE)

    num_vectors, dim = embeddings.shape

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Vektoren im Index: {index.ntotal}")

    faiss.write_index(index, str(config.INDEX_FILE))
    print(f"\nIndex gespeichert in: {config.INDEX_FILE}")

if __name__ == "__main__":
    main()