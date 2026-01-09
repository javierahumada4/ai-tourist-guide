import sys, json, numpy as np
from PIL import Image
import torch, open_clip
from ai_guide.pois import load_pois
from ai_guide.guide import make_guide_text

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model.eval()

pois = load_pois("data/pois")
emb = np.load("data/index/embeddings.npy")
ids = json.load(open("data/index/poi_ids.json"))

img = Image.open(sys.argv[1]).convert("RGB")
x = preprocess(img).unsqueeze(0)
with torch.no_grad():
    q = model.encode_image(x)
    q = q / q.norm(dim=-1, keepdim=True)
q = q.squeeze(0).numpy()

scores = emb @ q
best = ids[int(np.argmax(scores))]

print(make_guide_text(pois[best]))
