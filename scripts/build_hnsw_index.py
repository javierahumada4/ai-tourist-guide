import os
import json
import numpy as np
import hnswlib

INDEX_DIR = "data/index"

# HNSW parameters (good defaults; you can tune later)
M = 16                  # graph connectivity (bigger = more accurate, larger index)
EF_CONSTRUCTION = 200   # construction quality (bigger = better, slower build)

def main():
    emb_path = os.path.join(INDEX_DIR, "embeddings.npy")
    ids_path = os.path.join(INDEX_DIR, "poi_ids.json")
    out_index_path = os.path.join(INDEX_DIR, "hnsw_cosine.bin")
    out_meta_path = os.path.join(INDEX_DIR, "hnsw_cosine_meta.json")

    if not os.path.isfile(emb_path) or not os.path.isfile(ids_path):
        raise SystemExit("Missing embeddings. Run: python scripts/build_index_simple.py")

    emb = np.load(emb_path).astype(np.float32)  # (N, d), should be L2-normalized
    ids = json.load(open(ids_path, "r", encoding="utf-8"))

    if emb.ndim != 2:
        raise SystemExit("embeddings.npy must be 2D array (N, d).")

    n, dim = emb.shape
    if len(ids) != n:
        raise SystemExit("poi_ids.json length must match embeddings rows.")

    # HNSWlib index: use 'ip' (inner product). With normalized vectors, IP == cosine similarity.
    index = hnswlib.Index(space="ip", dim=dim)

    # init_index: max_elements must be >= n
    index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=M)

    # Add items with integer labels [0..n-1]
    labels = np.arange(n)
    index.add_items(emb, labels)

    # Optional: set ef for query time defaults (can be overridden at search time)
    index.set_ef(50)

    # Save index
    index.save_index(out_index_path)

    # Save metadata for reproducibility/debugging
    meta = {
        "space": "ip",
        "dim": dim,
        "count": n,
        "M": M,
        "ef_construction": EF_CONSTRUCTION,
        "note": "Vectors are expected to be L2-normalized so ip equals cosine similarity."
    }
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved HNSW index: {out_index_path}")
    print(f"[OK] Saved metadata:   {out_meta_path}")
    print(f"Indexed vectors: {n} | dim: {dim}")

if __name__ == "__main__":
    main()
