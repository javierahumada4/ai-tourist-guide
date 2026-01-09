import sys
import json
import numpy as np
from PIL import Image
import torch
import open_clip

from ai_guide.pois import load_pois
from ai_guide.guide import make_guide_text

# -----------------------
# Configuration
# -----------------------
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

TOP_K = 5

# Confidence gating
# These are starting values. We'll calibrate later once you have more POIs.
MIN_SCORE_CONFIDENT = 0.26      # if top1 < this -> UNKNOWN
MIN_MARGIN_CONFIDENT = 0.03     # if (top1 - top2) < this -> AMBIGUOUS

# -----------------------
# Helpers
# -----------------------
def embed_image(image_path: str, model, preprocess) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        q = model.encode_image(x)
        q = q / q.norm(dim=-1, keepdim=True)  # normalize for cosine similarity

    return q.squeeze(0).cpu().numpy().astype(np.float32)

def topk_cosine(query_vec: np.ndarray, mat: np.ndarray, k: int) -> list[tuple[int, float]]:
    # mat and query are normalized => cosine similarity = dot product
    scores = mat @ query_vec
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]

def classify_confidence(top1: float, top2: float | None) -> str:
    """
    Returns one of: CONFIDENT, AMBIGUOUS, UNKNOWN
    """
    if top1 < MIN_SCORE_CONFIDENT:
        return "UNKNOWN"
    if top2 is not None and (top1 - top2) < MIN_MARGIN_CONFIDENT:
        return "AMBIGUOUS"
    return "CONFIDENT"

# -----------------------
# Main
# -----------------------
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

    # Load index
    try:
        emb = np.load("data/index/embeddings.npy").astype(np.float32)
        ids = json.load(open("data/index/poi_ids.json", "r", encoding="utf-8"))
    except FileNotFoundError:
        print("Index not found. Run: python scripts/build_index_simple.py")
        sys.exit(1)

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval()

    # Query
    q = embed_image(image_path, model, preprocess)
    candidates = topk_cosine(q, emb, TOP_K)

    # Print Top-K
    print("\nTop candidates:")
    for rank, (i, score) in enumerate(candidates, start=1):
        pid = ids[i]
        name = pois.get(pid, {}).get("name", pid)
        print(f"{rank:>2}. {name:<40}  id={pid:<25}  score={score:.4f}")

    # Confidence decision
    top1_id = ids[candidates[0][0]]
    top1_score = candidates[0][1]
    top2_score = candidates[1][1] if len(candidates) > 1 else None

    status = classify_confidence(top1_score, top2_score)

    print("\nConfidence:", status)
    if top2_score is not None:
        print(f"Top1 score: {top1_score:.4f} | Top2 score: {top2_score:.4f} | Margin: {(top1_score - top2_score):.4f}")
    else:
        print(f"Top1 score: {top1_score:.4f}")

    # Decide what to output
    if status == "CONFIDENT":
        best_poi = pois[top1_id]
        print("\n" + "=" * 70)
        print(make_guide_text(best_poi))
    elif status == "AMBIGUOUS":
        print("\nResult is ambiguous. Suggested UX: ask the user to pick from the top candidates.")
    else:
        print("\nResult is unknown. Suggested UX: tell the user you are not sure and show top candidates.")

