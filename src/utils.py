# src/utils.py
import time
from typing import List

LABEL_MAP = {0: "negative", 1: "positive"}

def softmax_list(logits):
    import math
    exps = [math.exp(x) for x in logits]
    s = sum(exps)
    return [e/s for e in exps]

def format_results(preds: List[int], probs: List[List[float]]):
    out = []
    for p, pr in zip(preds, probs):
        out.append({"pred": int(p), "probs": [float(x) for x in pr], "label": LABEL_MAP.get(int(p), str(int(p)))})
    return out

def now_ms():
    return time.time() * 1000.0
