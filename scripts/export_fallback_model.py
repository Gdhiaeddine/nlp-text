"""
Export TF-IDF + Logistic Regression fallback model for MedStruct.
This allows the demo to run locally with solid performance (~0.76 macro-F1)
without requiring a GPU or large neural checkpoints.
"""

import os
import re
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

LABELS = ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'medstruct', 'data')
CKPT_DIR = os.path.join(BASE_DIR, 'medstruct', 'checkpoints')

def normalize_numbers(text: str) -> str:
    return re.sub(r'\d+', '@', str(text))

def parse_pubmed_rct(path: str) -> pd.DataFrame:
    rows, aid, sents = [], None, []
    def flush():
        n = len(sents)
        for i, (lab, txt) in enumerate(sents):
            rows.append((aid, i, n, lab, normalize_numbers(txt)))
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('###'):
                if sents: flush()
                aid, sents = line[3:].strip(), []
            elif line.strip() == '':
                if sents: flush(); sents = []
            else:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    sents.append((parts[0], parts[1]))
    if sents: flush()
    return pd.DataFrame(rows, columns=['abstract_id', 'sent_idx', 'n_sents', 'label', 'text'])

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, 'train.txt')
    dev_path = os.path.join(DATA_DIR, 'dev.txt')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")

    print("Loading PubMed 20k train dataset...")
    train_df = parse_pubmed_rct(train_path)
    print(f"Loaded {len(train_df):,} training sentences across {train_df['abstract_id'].nunique():,} abstracts.")

    print("Fitting TfidfVectorizer (ngram_range=(1,2), max_features=100000)...")
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100000, sublinear_tf=True)
    X_train = vec.fit_transform(train_df['text'])
    print(f"Vectorized train set in {time.time() - t0:.1f}s, shape: {X_train.shape}")

    print("Fitting Logistic Regression (C=3.0, max_iter=1000)...")
    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, C=3.0, random_state=1, solver='lbfgs')
    clf.fit(X_train, train_df['label'])
    print(f"Trained LogisticRegression in {time.time() - t0:.1f}s")

    # Evaluate on dev set if available
    if os.path.exists(dev_path):
        print("Evaluating on dev dataset...")
        dev_df = parse_pubmed_rct(dev_path)
        X_dev = vec.transform(dev_df['text'])
        p_dev = clf.predict(X_dev)
        macro_f1 = f1_score(dev_df['label'], p_dev, labels=LABELS, average='macro')
        print(f"Dev Macro-F1: {macro_f1:.4f}")
        print("\nClassification Report (Dev):")
        print(classification_report(dev_df['label'], p_dev, labels=LABELS, digits=4))

    # Save artifacts
    model_path = os.path.join(CKPT_DIR, 'tfidf_baseline.joblib')
    vec_path = os.path.join(CKPT_DIR, 'tfidf_vec.joblib')

    joblib.dump(clf, model_path)
    joblib.dump(vec, vec_path)
    print(f"Saved fallback model to: {model_path}")
    print(f"Saved vectorizer to: {vec_path}")

if __name__ == '__main__':
    main()
