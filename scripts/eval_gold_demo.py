"""
Evaluate the MedStruct Demo Prediction Pipeline on Gold Standard Abstracts.
Matches Step 6: "The demo only counts as evidence if you evaluate it."
Outputs Macro-F1, per-class metrics, and saves misclassifications for error analysis.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import argparse
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix

try:
    from app.predictor import predict_abstract, LABELS, normalize_numbers
except ImportError:
    from predictor import predict_abstract, LABELS, normalize_numbers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'medstruct', 'data')
RES_DIR = os.path.join(BASE_DIR, 'medstruct', 'results')

def evaluate_on_dataset(csv_path: str, output_error_audit: bool = True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Evaluation file not found at: {csv_path}")

    print(f"Loading evaluation dataset from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'abstract_id': str})
    
    # Validate expected columns
    for col in ['abstract_id', 'sent_idx', 'label', 'text']:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    df['label'] = df['label'].str.upper().str.strip()
    
    # Filter to compatible labels
    valid_mask = df['label'].isin(LABELS)
    if not valid_mask.all():
        ignored = df[~valid_mask]['label'].value_counts().to_dict()
        print(f"Filtering out non-canonical labels: {ignored}")
        df = df[valid_mask].copy()

    abstract_groups = df.groupby('abstract_id', sort=False)
    print(f"Evaluating {len(df):,} sentences across {len(abstract_groups):,} abstracts...")

    all_preds = []
    all_confs = []
    all_golds = []
    rows_for_audit = []

    for aid, grp in abstract_groups:
        full_text = " ".join(grp['text'].tolist())
        # Run prediction through the exact same demo pipeline
        res = predict_abstract(full_text)
        pred_df = res['df']
        
        # Check alignment
        if len(pred_df) == len(grp):
            for gold_row, (_, pred_row) in zip(grp.iterrows(), pred_df.iterrows()):
                gold_label = gold_row[1]['label']
                pred_label = pred_row['role']
                conf = pred_row['confidence']
                text_sent = gold_row[1]['text']
                
                all_golds.append(gold_label)
                all_preds.append(pred_label)
                all_confs.append(conf)

                if gold_label != pred_label:
                    rows_for_audit.append({
                        'abstract_id': aid,
                        'sent_idx': gold_row[1]['sent_idx'],
                        'gold_label': gold_label,
                        'pred_label': pred_label,
                        'confidence': conf,
                        'text': text_sent
                    })
        else:
            # Fallback sentence-by-sentence if split counts differ
            for _, r in grp.iterrows():
                res_sent = predict_abstract(r['text'])
                gold_label = r['label']
                if not res_sent['df'].empty:
                    pred_label = res_sent['df'].iloc[0]['role']
                    conf = res_sent['df'].iloc[0]['confidence']
                else:
                    pred_label = 'BACKGROUND'
                    conf = 0.20

                all_golds.append(gold_label)
                all_preds.append(pred_label)
                all_confs.append(conf)

                if gold_label != pred_label:
                    rows_for_audit.append({
                        'abstract_id': aid,
                        'sent_idx': r['sent_idx'],
                        'gold_label': gold_label,
                        'pred_label': pred_label,
                        'confidence': conf,
                        'text': r['text']
                    })

    # Metrics
    macro_f1 = f1_score(all_golds, all_preds, labels=LABELS, average='macro', zero_division=0)
    print("\n" + "=" * 50)
    print(f" DEMO EVALUATION RESULT on {os.path.basename(csv_path)}")
    print(f" Overall Macro-F1: {macro_f1:.4f}")
    print("=" * 50)
    print("\nDetailed Classification Report:")
    print(classification_report(all_golds, all_preds, labels=LABELS, digits=4, zero_division=0))

    if output_error_audit and rows_for_audit:
        os.makedirs(RES_DIR, exist_ok=True)
        audit_df = pd.DataFrame(rows_for_audit)
        audit_path = os.path.join(RES_DIR, f'demo_errors_{os.path.splitext(os.path.basename(csv_path))[0]}.csv')
        audit_df.to_csv(audit_path, index=False)
        print(f"\nSaved {len(rows_for_audit):,} misclassified sentences for error analysis to:\n  {audit_path}")

    return macro_f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate MedStruct Demo on Gold/Test Abstracts")
    parser.add_argument('--path', type=str, default=os.path.join(DATA_DIR, 'gold_pass_final.csv'),
                        help="Path to evaluation CSV file")
    args = parser.parse_args()

    if os.path.exists(args.path):
        evaluate_on_dataset(args.path)
    else:
        print(f"Gold set file not found at: {args.path}")
        print("To generate the gold set annotation sheet, run: python scripts/generate_annotation_sheet.py")

if __name__ == '__main__':
    main()
