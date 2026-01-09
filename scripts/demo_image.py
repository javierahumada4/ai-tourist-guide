import sys
import json
import numpy as np
from PIL import Image
import torch
import open_clip
import hnswlib

from ai_guide.pois import load_pois
from ai_guide.guide import make_guide_text

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

TOP_K = 3

# Confidence gating (starting points; calibrate later)
MIN_SCORE_CONFIDENT = 0.26
MIN_MARGIN_CONFIDENT = 0.03

# HNSW query parameter (bigger = more accurate, slower)
HNSW_EF_SEARCH = 200

def embed_image(image_path: str, model, preprocess) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        q = model.encode_image(x)
        q = q / q.norm(dim=-1, keepdim=True)
    return q.squeeze(0).cpu().numpy().astype(np.float32)

def classify_confidence(top1: float, top2: float | None) -> str:
    if top1 < MIN_SCORE_CONFIDENT:
        return "UNKNOWN"
    if top2 is not None and (top1 - top2) < MIN_MARGIN_CONFIDENT:
        return "AMBIGUOUS"
    return "CONFIDENT"

def load_hnsw(index_path: str, dim: int) -> hnswlib.Index:
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(index_path)
    idx.set_ef(max(HNSW_EF_SEARCH, TOP_K * 10))
    return idx

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/demo_image.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load POIs
    pois = load_pois("data/pois")
    if not pois:
        print("No POIs found in data/pois. Copy examples from docs/examples/pois first.")
        sys.exit(1)

    # Load IDs + embeddings (we need embeddings only for dim; index contains data already)
    ids = json.load(open("data/index/poi_ids.json", "r", encoding="utf-8"))
    emb = np.load("data/index/embeddings.npy").astype(np.float32)
    n, dim = emb.shape

    # Load HNSW index
    hnsw_path = "data/index/hnsw_cosine.bin"
    try:
        hnsw = load_hnsw(hnsw_path, dim)
    except Exception as e:
        print("HNSW index not found or failed to load.")
        print("Build it with: python scripts/build_hnsw_index.py")
        print("Error:", e)
        sys.exit(1)

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval()

    # Query embedding
    q = embed_image(image_path, model, preprocess)

    # HNSW query: returns (labels, distances)
    # For space='ip', distances returned by hnswlib are actually inner products (similarities)
    try:
        labels, distances = hnsw.knn_query(q, k=min(TOP_K, len(ids)))
    except RuntimeError as e:
        # Fallback: increase ef and retry once
        hnsw.set_ef(max(500, TOP_K * 50))
        labels, distances = hnsw.knn_query(q, k=min(TOP_K, len(ids)))
    
    labels = labels[0].tolist()
    distances = distances[0].tolist()

    # cosine distance = 1 - cosine similarity
    candidates = []
    for label, dist in zip(labels, distances):
        sim = 1.0 - float(dist)
        candidates.append((label, sim))

    print("\nTop candidates (HNSW cosine):")
    for rank, (label, sim) in enumerate(candidates, start=1):
        pid = ids[label]
        name = pois.get(pid, {}).get("name", pid)
        print(f"{rank:>2}. {name:<40}  id={pid:<25}  sim={sim:.4f}")

    top1_id = ids[candidates[0][0]]
    top1_score = candidates[0][1]
    top2_score = candidates[1][1] if len(candidates) > 1 else None
    status = classify_confidence(top1_score, top2_score)

    print("\nConfidence:", status)
    if top2_score is not None:
        print(f"Top1 score: {top1_score:.4f} | Top2 score: {top2_score:.4f} | Margin: {(top1_score - top2_score):.4f}")
    else:
        print(f"Top1 score: {top1_score:.4f}")

    if status == "CONFIDENT":
        print("\n" + "=" * 70)
        print(make_guide_text(pois[top1_id]))
    elif status == "AMBIGUOUS":
        print("\nResult is ambiguous. Suggested UX: ask the user to pick from the top candidates.")
    else:
        print("\nResult is unknown. Suggested UX: tell the user you are not sure and show top candidates.")
