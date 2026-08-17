from pathlib import Path

import pandas as pd
import streamlit as st

from utils import (
    load_bundle,
    analyse_text,
    count_words,
    MIN_DISPLAY_FEATURES,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "deployment_best_hybrid.joblib"

st.set_page_config(
    page_title="Hybrid AI Text Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1080px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.6rem 1.8rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 18px;
            margin-bottom: 1.1rem;
        }
        .hero h1 {
            margin: 0 0 .4rem 0;
            font-size: 2.25rem;
        }
        .hero p {
            margin: 0;
            opacity: .78;
            font-size: 1.05rem;
        }
        .result-card {
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
            height: 100%;
        }
        .result-label {
            font-size: .82rem;
            opacity: .68;
            text-transform: uppercase;
            letter-spacing: .07em;
            margin-bottom: .35rem;
        }
        .result-value {
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .note {
            border-left: 4px solid #7c8798;
            padding: .8rem 1rem;
            background: rgba(128,128,128,.07);
            border-radius: 8px;
            margin-top: 1rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.18);
            padding: .65rem .8rem;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle(MODEL_PATH)

try:
    bundle = get_bundle()
except Exception as exc:
    st.error(
        "The trained model bundle could not be loaded. "
        "Check that deployment_best_hybrid.joblib is present and that "
        "the package versions in requirements.txt are installed."
    )
    st.exception(exc)
    st.stop()

st.markdown(
    """
    <div class="hero">
        <h1>Hybrid AI-Generated Text Detection Framework</h1>
        <p>
            Analyse text using statistical, stylometric and semantic features
            with a validated Hybrid Linear SVM pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 0.75], gap="large")

with left:
    st.subheader("Text analysis")

    uploaded = st.file_uploader(
        "Optional: upload a UTF-8 .txt file",
        type=["txt"],
        help="A pasted text sample can be used instead.",
    )

    uploaded_text = ""
    if uploaded is not None:
        try:
            uploaded_text = uploaded.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("The uploaded file could not be decoded as UTF-8 text.")

    text = st.text_area(
        "Paste text to analyse",
        value=uploaded_text,
        height=300,
        placeholder="Paste at least 50 words of English text here...",
    )

    words = count_words(text)
    st.caption(f"Current word count: **{words:,}**")

    analyse_clicked = st.button(
        "Analyse text",
        type="primary",
        use_container_width=True,
    )

with right:
    st.subheader("Artefact")
    st.write("**Model:** Hybrid Linear SVM")
    st.write("**Semantic representation:** TF-IDF")
    st.write("**Engineered representation:** Statistical + stylometric")
    st.write(f"**Minimum recommended length:** {bundle['min_words']} words")
    st.write("**Training sources:** HC3 + DAIGT")

    with st.expander("Research validation summary", expanded=False):
        metrics = bundle.get("combined_test_metrics", {})
        if metrics:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Combined held-out accuracy", f"{metrics.get('Accuracy', 0):.2%}")
                st.metric("Combined held-out F1", f"{metrics.get('F1', 0):.2%}")
            with c2:
                st.metric("Combined ROC-AUC", f"{metrics.get('ROC-AUC', 0):.2%}")
                st.metric("Balanced accuracy", f"{metrics.get('Balanced Accuracy', 0):.2%}")

        st.markdown("**Cross-dataset stress testing**")
        st.write("HC3 → DAIGT F1: **54.68%**")
        st.write("DAIGT → HC3 F1: **65.53%**")
        st.caption(
            "The transfer tests show a substantial generalisation gap. "
            "The in-domain/combined figures must not be interpreted as universal accuracy."
        )

if analyse_clicked:
    if not text or not text.strip():
        st.warning("Please enter or upload text before running the analysis.")
    elif words < bundle["min_words"]:
        st.warning(
            f"This sample contains {words} words. "
            f"The model was designed for samples of at least {bundle['min_words']} words. "
            "Please provide a longer sample."
        )
    else:
        try:
            with st.spinner("Analysing text..."):
                result = analyse_text(text, bundle)

            st.divider()
            st.subheader("Prediction")

            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted class</div>
                        <div class="result-value">{result['prediction_name']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Model confidence</div>
                        <div class="result-value">{result['confidence']:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Words analysed</div>
                        <div class="result-value">{result['word_count']:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="note">
                    <strong>Important:</strong> This is a probabilistic research prediction.
                    Writing style, topic, editing and dataset differences can affect the result.
                    It should not be used as definitive evidence that a text was or was not
                    written by AI.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Selected feature analysis")
            feature_df = pd.DataFrame(
                [
                    {
                        "Feature": label,
                        "Value": result["display_features"][key],
                    }
                    for key, label in MIN_DISPLAY_FEATURES
                ]
            )
            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Technical details", expanded=False):
                p_human = result["probabilities"].get("Human", 0.0)
                p_ai = result["probabilities"].get("AI-generated", 0.0)

                st.write(f"**Human probability:** {p_human:.2%}")
                st.write(f"**AI-generated probability:** {p_ai:.2%}")
                st.write(f"**Feature condition:** {bundle['feature_condition']}")
                st.write(f"**Classifier:** {bundle['model_name']}")
                st.write(
                    "**TF-IDF settings:** "
                    f"max_features={bundle['tfidf_settings']['max_features']}, "
                    f"ngram_range={bundle['tfidf_settings']['ngram_range']}"
                )

        except Exception as exc:
            st.error("The analysis could not be completed.")
            st.exception(exc)

st.divider()
st.caption(
    "MSc research artefact — Hybrid Machine Learning Framework for Detecting AI-Generated Text. "
    "The interface is a demonstration layer for the validated research pipeline."
)
