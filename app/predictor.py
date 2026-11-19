"""
MedStruct Predictor Module.
Integrates with the medstruct_nlp_v3 (1) precision pipeline:
- Biomedical sentence splitting with abbreviation shielding.
- PubMed RCT number normalization (digits -> @).
- Active base model execution (Transformer BioBERT/SciBERT, Hierarchical BioBERT, or TF-IDF).
- medstruct_nlp_v3 (1) Order-Aware Markov Transition Decoding (Forward-Backward algorithm).
- medstruct_nlp_v3 (1) Temperature Scaling Calibration.
- medstruct_nlp_v3 (1) Calibrated Selective Abstention (Target 90% Precision).
"""

import os
import sys
from pathlib import Path
import re
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from app.model_loader import load_active_model, LABELS
    from app.precision_toolkit import (
        load_precision_config,
        apply_temperature,
        posterior_marginals,
        apply_thresholds,
    )
except ImportError:
    from model_loader import load_active_model, LABELS
    from precision_toolkit import (
        load_precision_config,
        apply_temperature,
        posterior_marginals,
        apply_thresholds,
    )

ABBREV = r'(?:e\.g|i\.e|et al|vs|cf|ca|no|approx|Fig|Figs|Eq|Ref|pp|Dr|Prof|Mr|Ms|Mrs)'

def split_sentences(text: str) -> List[str]:
    """
    Split an abstract into sentences while protecting common biomedical and
    scholarly abbreviations.
    Matches medstruct_nlp_v3 (1).ipynb cell 36.
    """
    if not text or not str(text).strip():
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
    Matches medstruct_nlp_v3 (1).ipynb cell 36.
    """
    return re.sub(r'\d+', '@', str(text))

def predict_abstract(
    text: str,
    threshold: float = 0.60,
    model_key: str = 'transformer',
    use_v3_decoding: bool = True,
    use_v3_calibration: bool = True,
    use_calibrated_thresholds: bool = True,
    custom_target_precision: Optional[float] = None
) -> Dict[str, Any]:
    """
    Given raw abstract text, split sentences, execute the active model,
    apply the medstruct_nlp_v3 (1) precision pipeline (temperature calibration,
    order-aware Markov decoding, and calibrated abstention), and return structured outputs.
    """
    sentences = split_sentences(text)
    if not sentences:
        return {
            'sentences': [],
            'df': pd.DataFrame(),
            'model_info': None,
            'grouped': {label: [] for label in LABELS + ['NEEDS REVIEW']},
            'v3_applied': False,
            'v3_config': None
        }

    # 1. Base Model Inference
    model_runner, model_info = load_active_model(model_key=model_key)
    normalized_sents = [normalize_numbers(s) for s in sentences]
    raw_probs = model_runner(normalized_sents)  # shape: (N, 5)

    # 2. Load medstruct_nlp_v3 (1) Precision Config
    cfg = load_precision_config()
    probs = raw_probs.copy()

    # Step 2a: Temperature Calibration from v3 (1)
    if use_v3_calibration and 'T' in cfg:
        calibrated_probs = apply_temperature(probs, cfg['T'])
    else:
        calibrated_probs = probs

    # Step 2b: Order-Aware Markov Sequence Decoding from v3 (1)
    if use_v3_decoding and 'logT' in cfg and 'logS' in cfg and 'logP' in cfg:
        abstract_ids = np.zeros(len(sentences), dtype=int)
        marginals = posterior_marginals(
            calibrated_probs,
            abstract_ids=abstract_ids,
            logT=cfg['logT'],
            logStart=cfg['logS'],
            logPrior=cfg['logP'],
            lam=cfg.get('lam', 0.75),
            beta=cfg.get('beta', 0.5)
        )
    else:
        marginals = calibrated_probs

    # Raw model guess (before sequence order decoding)
    raw_pred_indices = raw_probs.argmax(axis=1)
    raw_roles = [LABELS[i] for i in raw_pred_indices]

    # Decoded model guess (after sequence order decoding)
    decoded_indices = marginals.argmax(axis=1)
    decoded_roles = [LABELS[i] for i in decoded_indices]
    decoded_confs = marginals.max(axis=1)

    # Step 2c: Calibrated Abstention Thresholds from v3 (1)
    if use_calibrated_thresholds and 'th' in cfg:
        th = cfg['th']
        effective_roles_raw = apply_thresholds(marginals, th)
        # Convert 'UNSURE' to 'NEEDS REVIEW' for consistent UI display
        effective_roles = ['NEEDS REVIEW' if r == 'UNSURE' else r for r in effective_roles_raw]
        needs_review_flags = [bool(r == 'NEEDS REVIEW') for r in effective_roles]
    else:
        # Fallback to slider threshold
        effective_roles = []
        needs_review_flags = []
        for role, conf in zip(decoded_roles, decoded_confs):
            is_unsure = bool(conf < threshold)
            needs_review_flags.append(is_unsure)
            effective_roles.append('NEEDS REVIEW' if is_unsure else role)

    rows = []
    for idx in range(len(sentences)):
        sent = sentences[idx]
        raw_r = raw_roles[idx]
        dec_r = decoded_roles[idx]
        eff_r = effective_roles[idx]
        conf = float(decoded_confs[idx])
        is_rev = needs_review_flags[idx]
        order_corrected = bool(raw_r != dec_r)

        row = {
            'index': idx + 1,
            'sentence': sent,
            'role': dec_r,
            'effective_role': eff_r,
            'raw_role': raw_r,
            'order_corrected': order_corrected,
            'confidence': float(round(conf, 4)),
            'confidence_pct': float(round(conf * 100, 1)),
            'needs_review': is_rev,
        }
        for c_idx, label in enumerate(LABELS):
            row[f'prob_{label}'] = float(round(marginals[idx, c_idx], 4))
            row[f'raw_prob_{label}'] = float(round(raw_probs[idx, c_idx], 4))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Grouped structured abstract dictionary
    grouped: Dict[str, List[str]] = {label: [] for label in LABELS + ['NEEDS REVIEW']}
    for _, r in df.iterrows():
        grouped[r['effective_role']].append(r['sentence'])

    return {
        'sentences': sentences,
        'df': df,
        'model_info': model_info,
        'grouped': grouped,
        'v3_applied': bool(use_v3_decoding or use_calibrated_thresholds),
        'v3_config': cfg
    }
