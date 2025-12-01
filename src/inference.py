# src/inference.py
import os
from typing import List, Tuple, Optional, Dict
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

MODEL_DIR_ENV = "MODEL_DIR"
BASE_MODEL_ENV = "BASE_MODEL"

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_dir: str = None, base_model: str = None, merged_ok: bool = True):
    """
    Loads tokenizer and model. Supports PEFT adapter folder or merged model folder.
    model_dir: path to your student_model folder (tokenizer + adapter or merged model).
    base_model: AutoModel base id (used when adapter files are present).
    """
    model_dir = model_dir or os.getenv(MODEL_DIR_ENV, "./model/student_model")
    base_model = base_model or os.getenv(BASE_MODEL_ENV, "NousResearch/Llama-3.2-1B")
    device = get_device()

    # load tokenizer from saved artifact
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    # attempt to load a merged model first
    model = None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch.float16)
    except Exception:
        # fallback: load base model then attach PEFT adapter
        base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
        base.config.use_cache = False
        try:
            model = PeftModel.from_pretrained(base, model_dir)
        except Exception as e:
            # last resort: try loading as regular HF model from base and hope weights in adapter path
            raise RuntimeError(f"Failed to load model/adapter from '{model_dir}': {e}")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()
    model.to(device)
    return tokenizer, model, device

def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)

def predict_texts(
    texts: List[str],
    tokenizer,
    model,
    device,
    batch_size: int = 16,
    max_length: int = 128,
    calibrate_T: Optional[float] = None,
    threshold: float = 0.5,
) -> Tuple[List[int], List[List[float]], np.ndarray]:
    """
    Run batched inference and return preds, probs, logits (numpy).
    - calibrate_T: if provided applies temperature scaling to logits before softmax.
    - threshold: if provided uses prob of class 1 to override argmax and produce binary preds.
    """
    all_logits = []
    model_device = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        toks = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        toks = {k: v.to(model_device) for k, v in toks.items()}
        with torch.no_grad():
            out = model(**toks)
            logits = out.logits.detach().cpu().numpy()
        all_logits.append(logits)
    logits = np.concatenate(all_logits, axis=0)
    if calibrate_T and calibrate_T != 1.0:
        logits = logits / float(calibrate_T)
    probs = softmax_np(logits)
    argmax_preds = np.argmax(probs, axis=1).tolist()
    if threshold is not None:
        # binary decision using prob of positive class (index 1)
        pos_probs = probs[:, 1]
        threshold_preds = (pos_probs >= threshold).astype(int).tolist()
        preds = threshold_preds
    else:
        preds = argmax_preds
    return preds, probs.tolist(), logits
