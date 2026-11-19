"""
MedStruct Model Loader.
Supports loading and switching between:
1. Best Neural Transformer (BioBERT / SciBERT fine-tuned for PubMed RCT) - Active Default!
2. Hierarchical BioBERT (Local deep checkpoint if present in medstruct/checkpoints)
3. TF-IDF + Logistic Regression Baseline
4. Position-only baseline
"""

import os
import glob
from typing import Callable, Tuple, Dict, Any, List
import numpy as np

LABELS = ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'medstruct', 'checkpoints')

_MODEL_CACHE: Dict[str, Tuple[Callable[[List[str]], np.ndarray], Dict[str, Any]]] = {}

def get_available_models() -> Dict[str, str]:
    """Returns mapping of user-facing model labels to internal model keys."""
    choices = {
        "🧬 BioBERT / SciBERT (Best Model — Deep Contextual)": "transformer",
        "⚙️ TF-IDF + Logistic Regression (Fast Baseline)": "tfidf",
    }
    hier_pts = glob.glob(os.path.join(CKPT_DIR, '*hier*.pt')) if os.path.exists(CKPT_DIR) else []
    if hier_pts:
        choices["🏆 Hierarchical BioBERT (Local Checkpoint)"] = "hierarchical"
    return choices

def _load_transformer_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_id = 'gubartz/cls_scibert_pubmed_rct'
    tok_id = 'allenai/scibert_scivocab_uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        tok = AutoTokenizer.from_pretrained(tok_id, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(tok_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)

    model.to(device)
    model.eval()

    id2l = {int(k): v.upper() for k, v in model.config.id2label.items()}
    label_to_model_idx = {v: k for k, v in id2l.items()}
    col_indices = [label_to_model_idx[l] for l in LABELS]

    @torch.no_grad()
    def predict_fn(sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, len(LABELS)))
        inputs = tok(sentences, padding=True, truncation=True, max_length=96, return_tensors='pt').to(device)
        logits = model(**inputs).logits
        raw_probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return raw_probs[:, col_indices]

    info = {
        'name': 'BioBERT / SciBERT (PubMed RCT fine-tuned)',
        'type': 'Deep Contextual Transformer (Best Model)',
        'device': str(device),
        'macro_f1_estimate': '~0.82 - 0.86 (PubMed 20k RCT)',
        'architecture': 'Transformer Encoder + Linear Classification Head'
    }
    return predict_fn, info

def _load_tfidf_model():
    import joblib
    model_path = os.path.join(CKPT_DIR, 'tfidf_baseline.joblib')
    vec_path = os.path.join(CKPT_DIR, 'tfidf_vec.joblib')

    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        raise RuntimeError("TF-IDF checkpoints missing. Run scripts/export_fallback_model.py")

    clf = joblib.load(model_path)
    vec = joblib.load(vec_path)
    class_order = [list(clf.classes_).index(l) for l in LABELS]

    def predict_fn(sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, len(LABELS)))
        x = vec.transform(sentences)
        return clf.predict_proba(x)[:, class_order]

    info = {
        'name': 'TF-IDF + Logistic Regression',
        'type': 'N-Gram Linear Classifier (Baseline)',
        'device': 'CPU',
        'macro_f1_estimate': '0.7599 (Test) / 0.7671 (Dev)',
        'architecture': '100,000 unigrams/bigrams + L-BFGS'
    }
    return predict_fn, info

def _load_hierarchical_model():
    hier_pts = glob.glob(os.path.join(CKPT_DIR, '*hier*.pt'))
    if not hier_pts:
        # Fall back to best transformer
        return _load_transformer_model()

    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from transformers import AutoTokenizer, AutoModel

    best_ckpt = hier_pts[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'dmis-lab/biobert-v1.1' if 'BioBERT' in best_ckpt else 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)

    class HierModel(nn.Module):
        def __init__(self, enc, hid=128, drop=0.2, c=len(LABELS)):
            super().__init__()
            self.enc = enc
            self.lstm = nn.LSTM(enc.config.hidden_size, hid, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(drop)
            self.out = nn.Linear(2 * hid, c)

        def forward(self, ids, mask, sm):
            b, s, _ = ids.shape
            cls = self.enc(input_ids=ids[sm], attention_mask=mask[sm]).last_hidden_state[:, 0]
            x = torch.zeros(b, s, cls.size(-1), device=cls.device, dtype=cls.dtype)
            x[sm] = cls
            x = self.drop(x)
            packed = pack_padded_sequence(x, sm.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=s)
            return self.out(self.drop(h))

    model = HierModel(encoder)
    state_dict = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    @torch.no_grad()
    def predict_fn(sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, len(LABELS)))
        enc = tokenizer(sentences, padding='max_length', truncation=True, max_length=96, return_tensors='pt')
        ids = enc['input_ids'].unsqueeze(0).to(device)
        mask = enc['attention_mask'].unsqueeze(0).to(device)
        sm = torch.ones((1, len(sentences)), dtype=torch.bool, device=device)
        logits = model(ids, mask, sm)
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        return probs

    info = {
        'name': os.path.basename(best_ckpt),
        'type': 'Hierarchical BioBERT (Local Checkpoint)',
        'device': str(device),
        'macro_f1_estimate': '0.7953 (Test) / 0.8462 (Dev)',
        'architecture': 'BioBERT Encoder + BiLSTM Contextual Head'
    }
    return predict_fn, info

def load_active_model(model_key: str = 'transformer') -> Tuple[Callable[[List[str]], np.ndarray], Dict[str, Any]]:
    """
    Loads and returns (predict_fn, model_info) for the specified model key:
    'transformer' (default best model), 'tfidf', or 'hierarchical'.
    """
    global _MODEL_CACHE
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    if model_key == 'tfidf':
        res = _load_tfidf_model()
    elif model_key == 'hierarchical':
        res = _load_hierarchical_model()
    else:
        # Default: Best Transformer
        res = _load_transformer_model()

    _MODEL_CACHE[model_key] = res
    return res
