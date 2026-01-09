import os, json, numpy as np
from PIL import Image
import torch, open_clip

POIS = "data/pois"
IMAGES = "data/images"
OUT = "data/index"

os.makedirs(OUT, exist_ok=True)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model.eval()

vecs, ids = [], []

for fn in os.listdir(POIS):
    poi = json.load(open(os.path.join(POIS, fn)))
    pid = poi["id"]
    folder = os.path.join(IMAGES, pid)
    if not os.path.isdir(folder):
        continue

    embeddings = []
    for img_fn in os.listdir(folder):
        img = Image.open(os.path.join(folder, img_fn)).convert("RGB")
        x = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            v = model.encode_image(x)
            v = v / v.norm(dim=-1, keepdim=True)
        embeddings.append(v.squeeze(0).numpy())

    vec = np.mean(embeddings, axis=0)
    vec = vec / np.linalg.norm(vec)

    vecs.append(vec)
    ids.append(pid)

np.save(f"{OUT}/embeddings.npy", np.array(vecs, dtype="float32"))
json.dump(ids, open(f"{OUT}/poi_ids.json", "w"))
