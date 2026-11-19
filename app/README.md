# MedStruct: Rhetorical Role Classification for Biomedical Abstracts

MedStruct is an NLP demo application that segments and structures clinical and scientific abstracts into standard rhetorical roles:
- **BACKGROUND**: Disease context, public health challenge, prior research gaps.
- **OBJECTIVE**: Stated aims, hypothesis, and scope of investigation.
- **METHODS**: Experimental design, trial participants, cohort selection, interventions, outcomes measured.
- **RESULTS**: Quantitative and statistical findings, clinical outcomes, adverse events.
- **CONCLUSIONS**: Clinical interpretation, implications, future directions.
- **⚠️ NEEDS REVIEW**: Sentences where prediction confidence falls below the dev-calibrated threshold.

---

## Precision Toolkit Pipeline (from `medstruct_nlp_v3 (1).ipynb`)

The Streamlit app directly integrates the 3-stage precision post-processing pipeline from `medstruct_nlp_v3 (1)`:
1. **Temperature Calibration ($T=1.0$)**: Eliminates model overconfidence on dev-set predictions.
2. **Order-Aware Markov Transition Decoding ($\lambda=0.75, \beta=0.5$)**: Uses forward-backward sequence smoothing over RCT discourse priors ($logT, logS, logP$) to correct misordered sentences.
3. **Calibrated Selective Abstention**: Derives empirical confidence thresholds per rhetorical role targeting $\ge 90\%$ precision on key classes (`OBJECTIVE`, `METHODS`, `RESULTS`), flagging uncertain predictions as `NEEDS REVIEW` instead of forcing incorrect labels.

---

## Architecture & Model Hierarchy

The application prioritizes models in the following order:
1. **Hierarchical BioBERT (`BioBERT_hier-full_f1.0_s*.pt`)**: Full-abstract contextual model using sentence encoder + bidirectional sequence modeling.
2. **Sentence-level BioBERT (`BioBERT_sent_f1.0_s*.pt`)**: Isolated sentence transformer.
3. **TF-IDF + Logistic Regression Baseline (`tfidf_baseline.joblib`)**: High-speed, CPU-friendly fallback achieving $\approx 0.76$ macro-F1 with zero GPU dependencies.
4. *BiLSTM smoke checkpoints ($F_1 \approx 0.27$) are intentionally rejected by safety guards.*

---

## Quickstart (Local Run)

### 1. Install Dependencies
```bash
pip install -r app/requirements.txt
```

### 2. Export the Fallback Baseline (if not already built)
```bash
python scripts/export_fallback_model.py
```

### 3. Launch the Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## Evaluating the Demo on Gold Standard Data

In accordance with empirical validation criteria, the demo prediction pipeline can be audited against human-annotated gold abstracts:

```bash
python scripts/eval_gold_demo.py --path medstruct/data/gold_pass_final.csv
```
This generates:
- Macro-$F_1$ and per-class precision/recall/F1 metrics.
- A CSV audit of all misclassifications in `medstruct/results/` for error analysis in your research paper.

---

## Free Cloud Deployment

### Option A: Hugging Face Spaces (Recommended)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Select **Streamlit** as the Space SDK.
3. Push the contents of the `app/` folder (along with `medstruct/checkpoints/` or host large model weights on HF Hub / Drive).
4. Set the app entry point to `app.py`.

### Option B: Streamlit Community Cloud
1. Push your repository to GitHub.
2. Log in to [Streamlit Cloud](https://share.streamlit.io).
3. Connect your repository, choose branch `main`, and specify file path `app/app.py`.

---

## Model Limitations & Clinical Disclaimer

> [!WARNING]
> **Domain Boundaries**: MedStruct was trained on the PubMed 20k RCT benchmark (Randomized Controlled Trials).
> - It performs with high precision on structured and semi-structured clinical trials.
> - While functional on unstructured abstracts, non-clinical, observational, or preclinical (animal/in vitro) studies may have atypical sentence ordering, resulting in lower confidence.
> - Always review sentences flagged with **⚠️ NEEDS REVIEW**. MedStruct is a research and drafting tool, not a diagnostic medical device.
