"""
MedStruct Predictor Module.
Handles:
- Rule-based sentence segmentation with biomedical abbreviation shielding.
- Number normalization matching PubMed 20k RCT distribution (digits -> @).
- Abstract-level inference, confidence scoring, and review-flagging.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import re
from typing import Dict, List, Any
import numpy as np
import pandas as pd

try:
    from app.model_loader import load_active_model, LABELS
except ImportError:
    from model_loader import load_active_model, LABELS

ABBREV = r'(?:e\.g|i\.e|et al|vs|cf|ca|no|approx|Fig|Figs|Eq|Ref|pp|Dr|Prof|Mr|Ms|Mrs)'

def split_sentences(text: str) -> List[str]:
    """
    Split an abstract into sentences while protecting common biomedical and
    scholarly abbreviations.
    """
    if not text or not text.strip():
        return []
    cleaned = ' '.join(str(text).split())
    # Protect abbreviations by swapping trailing dot with section sign §
    shielded = re.sub(rf'\b{ABBREV}\.', lambda m: m.group(0)[:-1] + '§', cleaned, flags=re.IGNORECASE)
    # Split on terminal punctuation followed by space and capital letter, digit, or opening quote/bracket
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9("\'\[])', shielded)
    sentences = [p.replace('§', '.').strip() for p in parts if p.strip()]
    return sentences

def normalize_numbers(text: str) -> str:
    """
    Replace integer/float digit sequences with '@' to mirror the PubMed RCT token distribution.
    """
    return re.sub(r'\d+', '@', str(text))

def predict_abstract(text: str, threshold: float = 0.60, model_key: str = 'transformer') -> Dict[str, Any]:
    """
    Given raw abstract text, split sentences, execute the active model,
    and return structured outputs with confidence and review flags.
    """
    sentences = split_sentences(text)
    if not sentences:
        return {
            'sentences': [],
            'df': pd.DataFrame(),
            'model_info': None,
            'grouped': {label: [] for label in LABELS + ['NEEDS REVIEW']}
        }

    model_runner, model_info = load_active_model(model_key=model_key)
    # Normalize sentences for the model
    normalized_sents = [normalize_numbers(s) for s in sentences]
    
    # Run prediction
    probs = model_runner(normalized_sents)  # shape: (N, 5)
    
    argmax_indices = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    
    rows = []
    for idx, (sent, norm_sent, pred_idx, conf) in enumerate(zip(sentences, normalized_sents, argmax_indices, max_probs)):
        role = LABELS[pred_idx]
        needs_review = bool(conf < threshold)
        row = {
            'index': idx + 1,
            'sentence': sent,
            'role': role,
            'confidence': float(round(conf, 4)),
            'confidence_pct': float(round(conf * 100, 1)),
            'needs_review': needs_review,
            'effective_role': 'NEEDS REVIEW' if needs_review else role,
        }
        for c_idx, label in enumerate(LABELS):
            row[f'prob_{label}'] = float(round(probs[idx, c_idx], 4))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Grouped structured abstract dictionary
    grouped: Dict[str, List[str]] = {label: [] for label in LABELS + ['NEEDS REVIEW']}
    for _, r in df.iterrows():
        effective = r['effective_role']
        grouped[effective].append(r['sentence'])

    return {
        'sentences': sentences,
        'df': df,
        'model_info': model_info,
        'grouped': grouped
    }
