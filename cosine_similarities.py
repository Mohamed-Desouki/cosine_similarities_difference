# By Mohamed Desouki
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Calculate cosine similarity between models")
parser.add_argument("a", type=str, help="Path to model a")
parser.add_argument("b", type=str, help="Path to model b")
parser.add_argument("--out", type=str, help="Output file name, without extension", default="cosine_similarity", required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
args = parser.parse_args()

def loadModelWeights(mPath):
    model = torch.load(mPath, map_location=args.device)
    try: theta = model["state_dict"]
    except: theta = model
    return theta

a, b = loadModelWeights(args.a), loadModelWeights(args.b)

cosine_similarities = []

for key in tqdm(a.keys(), desc="Calculating cosine similarity"):
    if key in b and a[key].size() == b[key].size():
        a_flat = a[key].view(-1).to(torch.float32)
        b_flat = b[key].view(-1).to(torch.float32)
        simab = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0))
        cosine_similarities.append((key, simab.item()))

with open(f"lesser({args.a.split('/')[-1]}-{args.b.split('/')[-1]}).txt", 'w') as f1, open(f"greater({args.a.split('/')[-1]}-{args.b.split('/')[-1]}).txt", 'w') as f2:
    for key, similarity in cosine_similarities:
        if similarity < 0.9:
            f1.write(f"{key}: {similarity}\n")
        else:
            f2.write(f"{key}: {similarity}\n")

print(f"Done! Cosine similarity written to lesser({args.a.split('/')[-1]}-{args.b.split('/')[-1]}).txt and greater({args.a.split('/')[-1]}-{args.b.split('/')[-1]}).txt.")
