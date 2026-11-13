"""
Helper script to generate annotation CSV sheets for unstructured abstracts (Pass 1 and Pass 2)
and compute inter-annotator / test-retest Cohen's Kappa.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import re
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score

try:
    from app.predictor import split_sentences, LABELS
except ImportError:
    from predictor import split_sentences, LABELS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'medstruct', 'data')

def create_annotation_template(raw_abstracts: list[str], output_path: str):
    """Generates an empty annotation template CSV from a list of raw abstracts."""
    rows = []
    for i, abs_text in enumerate(raw_abstracts):
        aid = f"abs_{i+1:03d}"
        sents = split_sentences(abs_text)
        for j, s in enumerate(sents):
            rows.append({
                'abstract_id': aid,
                'sent_idx': j,
                'text': s,
                'label': ''  # To be filled by human annotator with one of LABELS
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created annotation sheet with {len(rows)} sentences across {len(raw_abstracts)} abstracts at:\n  {output_path}")

def calculate_cohen_kappa(pass1_path: str, pass2_path: str, output_final_path: str = None):
    """Computes Cohen's kappa between two annotation passes and checks consistency."""
    a = pd.read_csv(pass1_path, dtype={'abstract_id': str})
    b = pd.read_csv(pass2_path, dtype={'abstract_id': str})
    
    assert (a['text'].values == b['text'].values).all(), "Pass 1 and Pass 2 must have identical sentence orders!"
    
    la = a['label'].str.upper().str.strip()
    lb = b['label'].str.upper().str.strip()
    
    kappa = cohen_kappa_score(la, lb, labels=LABELS)
    raw_agree = (la == lb).mean()
    
    print("=" * 45)
    print(f"Cohen's Kappa (κ): {kappa:.3f}")
    print(f"Raw Agreement:    {raw_agree * 100:.2f}%")
    print("=" * 45)
    
    ct = pd.crosstab(la, lb, rownames=['Pass 1'], colnames=['Pass 2']).reindex(index=LABELS, columns=LABELS, fill_value=0)
    print("\nConfusion Matrix between Passes:")
    print(ct)
    
    disagreements = a.assign(pass1=la, pass2=lb)[la != lb][['abstract_id', 'sent_idx', 'text', 'pass1', 'pass2']]
    print(f"\nTotal Disagreements: {len(disagreements)}")
    
    if output_final_path and len(disagreements) == 0:
        a.to_csv(output_final_path, index=False)
        print(f"Consensus gold set saved to: {output_final_path}")
    elif output_final_path:
        print("Resolve disagreements before exporting the final gold set.")

    return kappa, disagreements

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Annotation Sheet Generator & Kappa Evaluator")
    parser.add_argument('--kappa', action='store_true', help="Compute Cohen's Kappa between two passes")
    parser.add_argument('--pass1', type=str, default=os.path.join(DATA_DIR, 'gold_pass_1.csv'))
    parser.add_argument('--pass2', type=str, default=os.path.join(DATA_DIR, 'gold_pass_2.csv'))
    args = parser.parse_args()

    if args.kappa:
        calculate_cohen_kappa(args.pass1, args.pass2)
    else:
        print("Run with --kappa --pass1 <path1> --pass2 <path2> to compute Kappa.")
