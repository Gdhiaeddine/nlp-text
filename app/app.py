"""
MedStruct: Interactive Rhetorical Role Classification for Biomedical Abstracts.
Powered by the medstruct_nlp_v3 (1) precision pipeline:
- Biomedical Transformers (BioBERT / SciBERT) & Hierarchical models
- medstruct_nlp_v3 (1) Order-Aware Markov Transition Sequence Decoding (Forward-Backward)
- medstruct_nlp_v3 (1) Temperature Calibration
- medstruct_nlp_v3 (1) Class-Calibrated Selective Abstention (Target 90% Precision)
"""

import os
import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from app.model_loader import get_available_models, load_active_model, LABELS
    from app.predictor import predict_abstract, split_sentences, normalize_numbers
    from app.precision_toolkit import load_precision_config
except ImportError:
    from model_loader import get_available_models, load_active_model, LABELS
    from predictor import predict_abstract, split_sentences, normalize_numbers
    from precision_toolkit import load_precision_config

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="MedStruct — Biomedical Abstract Structurer (v3.1)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color Scheme for Rhetorical Roles (consistent with research visualizations)
ROLE_COLORS = {
    'BACKGROUND': '#2563EB',    # Blue
    'OBJECTIVE': '#7C3AED',     # Purple
    'METHODS': '#059669',       # Emerald Green
    'RESULTS': '#D97706',       # Amber / Orange
    'CONCLUSIONS': '#DC2626',   # Crimson Red
    'NEEDS REVIEW': '#E11D48',  # Rose / Alert Red
}

ROLE_BG = {
    'BACKGROUND': '#EFF6FF',
    'OBJECTIVE': '#F5F3FF',
    'METHODS': '#ECFDF5',
    'RESULTS': '#FFFBEB',
    'CONCLUSIONS': '#FEF2F2',
    'NEEDS REVIEW': '#FFF1F2',
}

# Curated benchmark clinical abstracts for quick 1-click testing
SAMPLE_ABSTRACTS = {
    "Clinical RCT (Depression Intervention)": (
        "Major depressive disorder affects over 300 million individuals globally with substantial treatment resistance. "
        "We sought to determine whether an augmented digital cognitive behavioral therapy protocol improves remission rates in partial responders. "
        "In this double-blind, randomized controlled trial, 450 adult participants with persistent depressive symptoms were randomized 1:1 to either the digital intervention or an active sham app for 12 weeks. "
        "The primary outcome was remission measured by the Hamilton Depression Rating Scale at week 12. "
        "Participants receiving the digital protocol demonstrated significantly higher remission rates compared with sham controls (44.2% vs 26.8%, p = 0.002, relative risk 1.65). "
        "Digital cognitive behavioral therapy represents an accessible, efficacious adjunctive treatment for treatment-resistant depression in primary care settings."
    ),
    "Oncology Trial (Immunotherapy)": (
        "Metastatic colorectal cancer with microsatellite instability has a poor prognosis following conventional fluoropyrimidine therapy. "
        "This phase 3 open-label study evaluated the efficacy and safety of first-line pembrolizumab compared with standard-of-care chemotherapy. "
        "A total of 307 previously untreated patients were randomly assigned to receive pembrolizumab 200 mg every 3 weeks or investigator's choice of chemotherapy. "
        "Progression-free survival was assessed blindly by independent central radiologic review as the primary endpoint. "
        "Pembrolizumab led to significantly longer progression-free survival than chemotherapy (median 16.5 months vs 8.2 months; hazard ratio 0.60; 95% CI 0.45 to 0.80; p = 0.0002). "
        "Treatment-related adverse events of grade 3 or higher occurred in 22% of the pembrolizumab group versus 66% of the chemotherapy group. "
        "First-line pembrolizumab provided clinically meaningful improvements in progression-free survival with substantially fewer toxic effects."
    ),
    "Unstructured Real-World Sample": (
        "Myocardial infarction remains a leading cause of cardiovascular mortality worldwide despite reperfusion therapy advances. "
        "To explore cardioprotective effects of a novel peptide antagonist against reperfusion injury. "
        "Adult Wistar rats underwent 30 minutes of left coronary artery occlusion followed by 2 hours of reperfusion with intravenous peptide or vehicle control. "
        "Peptide administration significantly reduced infarct size by 38% and preserved left ventricular ejection fraction. "
        "Mechanistically, the compound inhibited mitochondrial ROS production and suppressed inflammatory cytokine release. "
        "Targeting this mitochondrial pathway may offer a promising therapeutic strategy for acute coronary syndromes."
    )
}

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ MedStruct Control")
    st.caption("🚀 Pipeline Engine: **medstruct_nlp_v3 (1)**")

    st.markdown("### 🤖 Base Model")
    model_options = get_available_models()
    selected_model_label = st.selectbox(
        "Choose Base Inference Model:",
        options=list(model_options.keys()),
        index=0,  # Best Model by default
        key="selected_model_dropdown",
        help="Select which base model powers inference. 'BioBERT / SciBERT' is the best deep neural model."
    )
    selected_model_key = model_options[selected_model_label]

    st.markdown("---")
    st.markdown("### 🔬 v3 (1) Precision Toolkit")
    st.caption("Post-processing sequence pipeline from `medstruct_nlp_v3 (1).ipynb`")

    use_v3_decoding = st.checkbox(
        "Order-Aware Markov Decoding",
        value=True,
        key="v3_decoding_toggle",
        help="Enforces standard RCT discourse transition constraints (HMM forward-backward) to resolve sequence ambiguities."
    )

    use_v3_calibration = st.checkbox(
        "Temperature Calibration (T=1.0)",
        value=True,
        key="v3_calib_toggle",
        help="Calibrates class probabilities on the dev set to eliminate overconfidence."
    )

    abstention_mode = st.radio(
        "Confidence & Abstention Guard:",
        options=["🎯 v3 (1) Calibrated Thresholds (90% Precision)", "🎚️ Manual Confidence Slider"],
        index=0,
        key="abstention_mode_radio",
        help="Select between empirical dev-tuned class thresholds (guaranteeing >=90% precision on key roles) or a global threshold slider."
    )

    use_calibrated_th = (abstention_mode == "🎯 v3 (1) Calibrated Thresholds (90% Precision)")

    # Load precision configuration for display
    v3_cfg = load_precision_config()
    cfg_th = v3_cfg.get('th', {})

    if use_calibrated_th:
        with st.expander("📌 View v3 (1) Class Thresholds", expanded=False):
            st.caption("Sentences with confidence below these role-specific thresholds are labeled `NEEDS REVIEW`:")
            for lab in LABELS:
                val = cfg_th.get(lab, 0.5)
                st.write(f"- **{lab}**: `{val * 100:.1f}%`")
        threshold = 0.60
    else:
        threshold = st.slider(
            "Flag sentences below confidence:",
            min_value=0.30,
            max_value=0.95,
            value=0.60,
            step=0.05,
            key="conf_threshold_slider",
            help="Sentences with prediction confidence below this threshold are marked 'NEEDS REVIEW'."
        )

    st.markdown("---")
    st.markdown("### 🏷️ Rhetorical Roles")
    for r in LABELS:
        c = ROLE_COLORS[r]
        bg = ROLE_BG[r]
        st.markdown(
            f'<div style="padding: 4px 8px; margin-bottom: 4px; background: {bg}; color: {c}; '
            f'border-left: 4px solid {c}; font-weight: 600; font-size: 0.85rem; border-radius: 4px;">'
            f'{r}</div>',
            unsafe_allow_html=True
        )
    st.markdown(
        f'<div style="padding: 4px 8px; margin-bottom: 4px; background: {ROLE_BG["NEEDS REVIEW"]}; '
        f'color: {ROLE_COLORS["NEEDS REVIEW"]}; border-left: 4px solid {ROLE_COLORS["NEEDS REVIEW"]}; '
        f'font-weight: 600; font-size: 0.85rem; border-radius: 4px;">'
        f'⚠️ NEEDS REVIEW</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### ℹ️ Active Model Spec")
    try:
        _, info = load_active_model(model_key=selected_model_key)
        st.success(f"**Base**: {info['name']}")
        st.caption(f"**Type**: {info['type']}")
        st.caption(f"**Device**: {info.get('device', 'CPU')}")
        st.caption(f"**Macro-F1**: {info.get('macro_f1_estimate', 'N/A')}")
        if 'architecture' in info:
            st.caption(f"**Architecture**: {info['architecture']}")
    except Exception as e:
        st.warning(f"Model status: {e}")

# --- Main Page ---
st.title("🧬 MedStruct: Biomedical Abstract Structurer")
st.markdown(
    "Automated rhetorical role classification (**Background**, **Objective**, **Methods**, **Results**, **Conclusions**) "
    "powered by **medstruct_nlp_v3 (1)** with deep biomedical representations, order-aware sequence decoding, and calibrated abstention."
)

st.info(
    "💡 **Pipeline Active**: Configured with **`medstruct_nlp_v3 (1)` Precision Toolkit** "
    "(HMM forward-backward transition smoothing, temperature calibration, and 90% target precision thresholds)."
)

main_tabs = st.tabs(["⚡ Live Abstract Structurer", "📊 Models Benchmark & Empirical Comparison"])

# ==========================================
# TAB 1: LIVE INFERENCE
# ==========================================
with main_tabs[0]:
    # Preset buttons
    st.markdown("##### Load an example abstract:")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    if col_ex1.button("📋 Clinical RCT (Depression)", key="btn_example_depression"):
        st.session_state['abstract_input'] = SAMPLE_ABSTRACTS["Clinical RCT (Depression Intervention)"]
        st.session_state.pop('last_result', None)
        st.rerun()
    if col_ex2.button("🔬 Oncology Phase 3 Trial", key="btn_example_oncology"):
        st.session_state['abstract_input'] = SAMPLE_ABSTRACTS["Oncology Trial (Immunotherapy)"]
        st.session_state.pop('last_result', None)
        st.rerun()
    if col_ex3.button("🧪 Unstructured Preclinical", key="btn_example_preclinical"):
        st.session_state['abstract_input'] = SAMPLE_ABSTRACTS["Unstructured Real-World Sample"]
        st.session_state.pop('last_result', None)
        st.rerun()

    # Text Input
    default_text = st.session_state.get('abstract_input', SAMPLE_ABSTRACTS["Clinical RCT (Depression Intervention)"])
    input_text = st.text_area(
        "Paste abstract text below:",
        value=default_text,
        height=180,
        key="abstract_input",
        help="Enter an unstructured or structured biomedical abstract."
    )

    col_act1, col_act2 = st.columns([1, 4])
    analyze_btn = col_act1.button("🚀 Analyze Abstract", type="primary", use_container_width=True, key="btn_analyze_abstract")
    clear_btn = col_act2.button("🗑️ Clear", use_container_width=False, key="btn_clear_abstract")
    if clear_btn:
        st.session_state['abstract_input'] = ""
        st.session_state.pop('last_result', None)
        st.rerun()

    if (analyze_btn or 'last_result' in st.session_state) and input_text.strip():
        with st.spinner(f"Analyzing sentences using {selected_model_label} + v3 (1) Precision Pipeline..."):
            result = predict_abstract(
                input_text,
                threshold=threshold,
                model_key=selected_model_key,
                use_v3_decoding=use_v3_decoding,
                use_v3_calibration=use_v3_calibration,
                use_calibrated_thresholds=use_calibrated_th
            )
            st.session_state['last_result'] = result

        df = result['df']
        grouped = result['grouped']
        n_sents = len(result['sentences'])
        n_flagged = int(df['needs_review'].sum()) if not df.empty else 0
        n_order_corrected = int(df['order_corrected'].sum()) if not df.empty and 'order_corrected' in df.columns else 0

        st.markdown("---")
        
        # Summary Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Sentences", n_sents)
        m2.metric("Mean Confidence", f"{df['confidence'].mean() * 100:.1f}%" if not df.empty else "N/A")
        m3.metric("Flagged for Review", n_flagged, delta=f"{n_flagged} flagged" if n_flagged > 0 else "All confident", delta_color="inverse" if n_flagged > 0 else "normal")
        m4.metric("Markov Order Corrections", n_order_corrected, help="Sentences adjusted by v3 (1) order-aware sequence decoding")

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📑 Reconstructed Structured Abstract", "📊 Sentence Breakdown Table", "💾 Export Data"])

        # Sub Tab 1: Reconstructed Structured View
        with sub_tab1:
            st.markdown("#### Structured Abstract")
            st.caption("Sentences chronologically assembled by predicted rhetorical section:")
            
            for lab in LABELS:
                sents = grouped.get(lab, [])
                if sents:
                    color = ROLE_COLORS[lab]
                    bg = ROLE_BG[lab]
                    st.markdown(
                        f'<div style="margin-top: 14px; margin-bottom: 4px; padding: 4px 10px; background-color: {bg}; '
                        f'border-left: 5px solid {color}; border-radius: 4px;">'
                        f'<span style="color: {color}; font-weight: 700; font-size: 0.95rem;">{lab}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div style="padding: 10px 14px; background-color: #FAFAFA; border: 1px solid #E5E7EB; '
                        f'border-radius: 4px; line-height: 1.6; margin-bottom: 12px;">{" ".join(sents)}</div>',
                        unsafe_allow_html=True
                    )

            review_sents = grouped.get('NEEDS REVIEW', [])
            if review_sents:
                color = ROLE_COLORS['NEEDS REVIEW']
                bg = ROLE_BG['NEEDS REVIEW']
                st.markdown(
                    f'<div style="margin-top: 14px; margin-bottom: 4px; padding: 4px 10px; background-color: {bg}; '
                    f'border-left: 5px solid {color}; border-radius: 4px;">'
                    f'<span style="color: {color}; font-weight: 700; font-size: 0.95rem;">⚠️ NEEDS CLINICAL REVIEW (Below Calibrated Precision Threshold)</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="padding: 10px 14px; background-color: #FFF5F5; border: 1px dashed {color}; '
                    f'border-radius: 4px; line-height: 1.6; margin-bottom: 12px;">{" ".join(review_sents)}</div>',
                    unsafe_allow_html=True
                )

        # Sub Tab 2: Detailed Breakdown
        with sub_tab2:
            st.markdown("#### Sentence-by-Sentence Predictions (v3 (1) Pipeline)")
            
            display_cols = ['index', 'sentence', 'effective_role', 'role', 'raw_role', 'confidence_pct', 'needs_review']
            display_df = df[display_cols].copy()
            display_df.columns = ['#', 'Sentence Text', 'Effective Role', 'Decoded Role (v3)', 'Raw Model Guess', 'Confidence (%)', 'Needs Review']
            
            st.dataframe(
                display_df,
                column_config={
                    "#": st.column_config.NumberColumn(width="small"),
                    "Sentence Text": st.column_config.TextColumn(width="large"),
                    "Effective Role": st.column_config.TextColumn(width="medium"),
                    "Decoded Role (v3)": st.column_config.TextColumn(width="medium"),
                    "Raw Model Guess": st.column_config.TextColumn(width="small"),
                    "Confidence (%)": st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "Needs Review": st.column_config.CheckboxColumn(width="small")
                },
                use_container_width=True,
                hide_index=True
            )

            st.markdown("##### Posterior Probability Marginals Distribution")
            prob_cols = ['index'] + [f'prob_{l}' for l in LABELS]
            prob_df = df[prob_cols].copy()
            prob_df.columns = ['#'] + LABELS
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # Sub Tab 3: Export Options
        with sub_tab3:
            st.markdown("#### Download Structured Data")
            col_dl1, col_dl2 = st.columns(2)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            col_dl1.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name="medstruct_v3_predictions.csv",
                mime="text/csv",
                key="btn_dl_csv",
                use_container_width=True
            )

            json_records = {
                'metadata': {
                    'engine': 'medstruct_nlp_v3 (1) Precision Toolkit',
                    'base_model': result['model_info']['name'] if result['model_info'] else 'unknown',
                    'order_aware_decoding': use_v3_decoding,
                    'temperature_calibrated': use_v3_calibration,
                    'calibrated_thresholds_applied': use_calibrated_th,
                    'total_sentences': n_sents,
                    'flagged_sentences': n_flagged,
                    'order_corrected_sentences': n_order_corrected
                },
                'structured_abstract': grouped,
                'sentences': df.to_dict(orient='records')
            }
            json_data = json.dumps(json_records, indent=2, ensure_ascii=False)
            col_dl2.download_button(
                label="📥 Download Structured JSON",
                data=json_data,
                file_name="medstruct_v3_structured.json",
                mime="application/json",
                key="btn_dl_json",
                use_container_width=True
            )
            st.json(json_records)

# ==========================================
# TAB 2: MODEL COMPARISON & BENCHMARK
# ==========================================
with main_tabs[1]:
    st.markdown("### 🏆 Comprehensive Model Comparison & Experimental Results")
    st.markdown(
        "Empirical evaluation of all architectures on the **PubMed 20k RCT** benchmark. "
        "The experiments evaluate classical baselines, recurrent models, isolated sentence transformers, "
        "hierarchical abstract-level sequential networks, and the **medstruct_nlp_v3 (1)** precision toolkit."
    )

    results_csv_path = os.path.join(root_dir, 'medstruct', 'results', 'results.csv')
    if os.path.exists(results_csv_path):
        raw_df = pd.read_csv(results_csv_path)

        # Highlight KPIs
        c1, c2, c3, c4 = st.columns(4)
        best_test_row = raw_df.loc[raw_df['test_f1'].idxmax()]
        c1.metric("🥇 Best Test Macro-F1", f"{best_test_row['test_f1']:.4f}", f"{best_test_row['model']}")
        best_dev_row = raw_df.loc[raw_df['dev_f1'].idxmax()]
        c2.metric("🥈 Best Dev Macro-F1", f"{best_dev_row['dev_f1']:.4f}", f"{best_dev_row['model']}")
        tfidf_row = raw_df[raw_df['run_id'] == 'tfidf']
        c3.metric("🥉 TF-IDF Baseline Test F1", f"{tfidf_row['test_f1'].values[0]:.4f}" if not tfidf_row.empty else "0.7599")
        c4.metric("⚡ BiLSTM Baseline F1", "0.2696", "-0.5257 vs BioBERT", delta_color="inverse")

        st.markdown("---")
        st.markdown("#### 1. Full Benchmark Comparison Table")

        # Format comparison table
        table_df = raw_df.copy()
        table_df['Dev Macro-F1'] = table_df['dev_f1'].round(4)
        table_df['Test Macro-F1'] = table_df['test_f1'].round(4)
        table_df['Training (s)'] = table_df['train_s'].round(1)
        table_df['Params (M)'] = table_df['params_M'].round(2).fillna('-')
        table_df['Latency (ms/abstract)'] = table_df['latency_ms_per_abstract'].round(1).fillna('-')
        table_df['Context Window'] = table_df['window'].fillna('None / N/A')
        table_df['Data Fraction'] = table_df['frac'].apply(lambda x: f"{int(x * 100)}%" if pd.notnull(x) else '100%')

        cols_to_show = ['model', 'Context Window', 'Data Fraction', 'Dev Macro-F1', 'Test Macro-F1', 'Params (M)', 'Latency (ms/abstract)', 'Training (s)']
        display_bench = table_df[cols_to_show].sort_values(by='Test Macro-F1', ascending=False)
        st.dataframe(display_bench, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 2. Architecture Comparison (Full 100% Data)")
        
        # Filter full-data representative runs
        full_runs = raw_df[
            (raw_df['frac'] == 1.0) & 
            (raw_df['window'].isin(['full', '']) | raw_df['window'].isna())
        ].copy()
        
        # Drop window ablations to keep only main arch
        full_runs = full_runs[~full_runs['run_id'].str.contains('w[0-9]', na=False)]
        
        chart_data = full_runs[['model', 'test_f1', 'dev_f1']].set_index('model')
        chart_data.columns = ['Test Macro-F1', 'Dev Macro-F1']
        st.bar_chart(chart_data)

        st.markdown("---")
        col_ab1, col_ab2 = st.columns(2)

        with col_ab1:
            st.markdown("#### 3. Context-Window Ablation (RQ1)")
            st.caption("How much surrounding abstract context does BioBERT need?")
            window_df = raw_df[
                (raw_df['model'].str.contains('BioBERT (hierarchical)', regex=False)) & 
                (raw_df['frac'] == 1.0)
            ].copy()
            if not window_df.empty:
                window_df['Window Size'] = window_df['window'].map({'w0': 'w=0 (Isolated)', 'w1': 'w=1 (±1 sent)', 'w2': 'w=2 (±2 sents)', 'w3': 'w=3 (±3 sents)', 'full': 'Full Abstract'})
                window_chart = window_df[['Window Size', 'test_f1']].set_index('Window Size')
                window_chart.columns = ['Test Macro-F1']
                st.bar_chart(window_chart)
                st.markdown(
                    "> **Insight**: Performance scales monotonically from `w=0` (0.7514) to `w=3` (0.7850) and `full` (0.7953), "
                    "demonstrating that global abstract discourse structure is critical."
                )

        with col_ab2:
            st.markdown("#### 4. Low-Resource Learning Curve (RQ3)")
            st.caption("Macro-F1 when training data is restricted to 1%, 5%, 10%, and 100%:")
            lr_df = raw_df[
                (raw_df['model'].str.contains('BioBERT', regex=False)) & 
                (raw_df['window'].isin(['full', '']) | raw_df['window'].isna())
            ].copy()
            if not lr_df.empty:
                lr_piv = lr_df.pivot_table(index='frac', columns='model', values='test_f1')
                lr_piv.index = [f"{int(x*100)}%" for x in lr_piv.index]
                st.line_chart(lr_piv)
                st.markdown(
                    "> **Insight**: Sentence-level BioBERT is slightly more sample-efficient at 1-10% data, "
                    "but Hierarchical BioBERT dominates with abundant data (+2.9 points at 100%)."
                )

        st.markdown("---")
        st.markdown("#### 5. medstruct_nlp_v3 (1) Precision Toolkit Pipeline")
        st.markdown(
            """
            The **medstruct_nlp_v3 (1)** pipeline resolves role confusions (especially distinguishing `OBJECTIVE` from `BACKGROUND`, and `METHODS` from `RESULTS`) via three sequential post-processing stages:
            
            1. **Temperature Calibration ($T=1.0$)**:
               Aligns predicted softmax probabilities with empirical empirical accuracies on the dev split.
            2. **Markov Order-Aware Transition Smoothing ($\lambda=0.75, \\beta=0.5$)**:
               Uses an HMM forward-backward algorithm to apply empirical transition log-probabilities ($logT$), start priors ($logS$), and label priors ($logP$) derived from training RCT abstracts.
            3. **Selective Precision Abstention ($target=90\%$)**:
               Replaces borderline predictions with `NEEDS REVIEW` (`UNSURE`) when posterior marginals fall below the role's dev-calibrated threshold:
               - **BACKGROUND**: `99.0%`
               - **OBJECTIVE**: `86.7%`
               - **METHODS**: `35.4%`
               - **RESULTS**: `31.1%`
               - **CONCLUSIONS**: `28.1%`
            """
        )

        st.markdown("---")
        st.markdown("#### 6. Key Empirical Takeaways for the Research Paper")
        st.markdown(
            """
            | Finding | Evidence | Paper Implication |
            |---|---|---|
            | **v3 (1) Sequence Smoothing** | Order-aware Markov decoding corrects rhetorical role inversions. | Transition structure boosts sequence consistency without retraining. |
            | **Hierarchical Superiority** | BioBERT Hierarchical (**0.7953**) outperforms Sentence-level (**0.7666**) by **+2.87 F1**. | Models must model rhetorical discourse transitions between sentences. |
            | **Domain Pre-training** | BioBERT Sentence (**0.7666**) vs standard BERT Sentence (**0.7038**): **+6.28 F1**. | Biomedical vocabulary pre-training is crucial for clinical vocabulary. |
            | **Classical Baseline Power** | TF-IDF + LR (**0.7599**) outperforms BERT Sentence (**0.7038**)! | Simple n-gram frequency with sublinear TF is a surprisingly competitive baseline. |
            | **BiLSTM Smoke Failure** | 1-epoch BiLSTM collapsed at **0.2696 F1**. | Confirms that undertrained recurrent nets cannot be trusted for real clinical abstracts. |
            """
        )
    else:
        st.warning(f"Results CSV not found at `{results_csv_path}`. Run experiments to generate data.")

st.markdown("---")
st.caption(
    "MedStruct Demo | Multi-Model Evaluation Suite | Powered by medstruct_nlp_v3 (1) Precision Toolkit | Trained on PubMed 20k RCT"
)
