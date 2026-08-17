from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import textstat


FUNCTION_WORDS = {
    "the","a","an","and","or","but","if","while","because","although","as",
    "of","to","in","on","at","for","from","with","without","by","about",
    "into","through","during","before","after","above","below","is","am",
    "are","was","were","be","been","being","do","does","did","have","has",
    "had","i","me","my","mine","we","us","our","ours","you","your","yours",
    "he","him","his","she","her","hers","it","its","they","them","their",
    "theirs","this","that","these","those","who","whom","which","what"
}

FIRST_PERSON = {"i","me","my","mine","we","us","our","ours"}
SECOND_PERSON = {"you","your","yours","yourself","yourselves"}

STAT_FEATURES = [
    "word_count",
    "char_count",
    "sentence_count",
    "avg_sentence_length",
    "sentence_length_std",
    "avg_word_length",
    "type_token_ratio",
    "hapax_ratio",
    "lexical_density",
    "flesch_reading_ease",
    "repeated_word_ratio",
]

STYLO_FEATURES = [
    "punctuation_ratio",
    "comma_ratio",
    "semicolon_ratio",
    "colon_ratio",
    "question_ratio",
    "exclamation_ratio",
    "quote_ratio",
    "uppercase_ratio",
    "digit_ratio",
    "function_word_ratio",
    "first_person_ratio",
    "second_person_ratio",
    "paragraph_count",
    "contraction_ratio",
]

ALL_ENGINEERED_FEATURES = STAT_FEATURES + STYLO_FEATURES

MIN_DISPLAY_FEATURES = [
    ("word_count", "Word count"),
    ("avg_sentence_length", "Average sentence length"),
    ("type_token_ratio", "Lexical diversity (type-token ratio)"),
    ("flesch_reading_ease", "Flesch reading ease"),
    ("repeated_word_ratio", "Repeated-word ratio"),
    ("punctuation_ratio", "Punctuation ratio"),
]


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text)))


def split_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [s for s in sentences if s.strip()]


def tokenize_words(text: str):
    return re.findall(r"\b[\w'-]+\b", str(text).lower())


def extract_engineered_features(text: str) -> Dict[str, float]:
    """
    Reproduces the engineered feature extraction used by the training notebook.

    IMPORTANT:
    Do not simplify or independently 'improve' this function without retraining
    the deployment bundle. The inference feature order must remain identical to
    the fitted StandardScaler.
    """
    text = str(text)
    words = tokenize_words(text)
    sentences = split_sentences(text)

    n_words = len(words)
    n_chars = len(text)
    n_sentences = len(sentences)

    safe_words = max(n_words, 1)
    safe_chars = max(n_chars, 1)

    sentence_lengths = [len(tokenize_words(s)) for s in sentences] or [0]
    word_lengths = [len(w) for w in words] or [0]

    unique_words = len(set(words))
    counts = pd.Series(words).value_counts() if words else pd.Series(dtype=float)
    hapax_count = int((counts == 1).sum()) if len(counts) else 0
    repeated_tokens = int((counts[counts > 1] - 1).sum()) if len(counts) else 0

    function_count = sum(w in FUNCTION_WORDS for w in words)
    first_person_count = sum(w in FIRST_PERSON for w in words)
    second_person_count = sum(w in SECOND_PERSON for w in words)

    punctuation_count = sum(ch in ".,;:!?\"'()-" for ch in text)

    try:
        readability = (
            textstat.flesch_reading_ease(text)
            if n_words >= 10
            else np.nan
        )
    except Exception:
        readability = np.nan

    features = {
        # Statistical
        "word_count": n_words,
        "char_count": n_chars,
        "sentence_count": n_sentences,
        "avg_sentence_length": float(np.mean(sentence_lengths)),
        "sentence_length_std": float(np.std(sentence_lengths)),
        "avg_word_length": float(np.mean(word_lengths)),
        "type_token_ratio": unique_words / safe_words,
        "hapax_ratio": hapax_count / safe_words,
        "lexical_density": (safe_words - function_count) / safe_words,
        "flesch_reading_ease": readability,
        "repeated_word_ratio": repeated_tokens / safe_words,

        # Stylometric
        "punctuation_ratio": punctuation_count / safe_chars,
        "comma_ratio": text.count(",") / safe_chars,
        "semicolon_ratio": text.count(";") / safe_chars,
        "colon_ratio": text.count(":") / safe_chars,
        "question_ratio": text.count("?") / safe_chars,
        "exclamation_ratio": text.count("!") / safe_chars,
        "quote_ratio": (text.count('"') + text.count("'")) / safe_chars,
        "uppercase_ratio": sum(ch.isupper() for ch in text) / safe_chars,
        "digit_ratio": sum(ch.isdigit() for ch in text) / safe_chars,
        "function_word_ratio": function_count / safe_words,
        "first_person_ratio": first_person_count / safe_words,
        "second_person_ratio": second_person_count / safe_words,
        "paragraph_count": max(
            1,
            len([p for p in re.split(r"\n\s*\n", text) if p.strip()]),
        ),
        "contraction_ratio": len(
            re.findall(r"\b\w+['’]\w+\b", text)
        ) / safe_words,
    }

    return features


def load_bundle(model_path: Path | str) -> Dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    bundle = joblib.load(model_path)

    required = {
        "feature_condition",
        "model_name",
        "model",
        "transformers",
        "all_engineered_features",
        "min_words",
        "tfidf_settings",
    }
    missing = required.difference(bundle.keys())
    if missing:
        raise ValueError(f"Deployment bundle is missing keys: {sorted(missing)}")

    if bundle["feature_condition"] != "Hybrid":
        raise ValueError(
            "This interface is configured for the Hybrid deployment bundle."
        )

    expected_features = list(bundle["all_engineered_features"])
    if expected_features != ALL_ENGINEERED_FEATURES:
        raise ValueError(
            "Feature definition mismatch between utils.py and the trained bundle. "
            "Do not deploy until the inference feature list exactly matches training."
        )

    transformers = bundle["transformers"]
    for key in ("scaler", "tfidf"):
        if key not in transformers:
            raise ValueError(f"Deployment bundle is missing transformer: {key}")

    return bundle


def build_hybrid_matrix(text: str, bundle: Dict[str, Any]):
    engineered = extract_engineered_features(text)

    feature_frame = pd.DataFrame(
        [[engineered[name] for name in ALL_ENGINEERED_FEATURES]],
        columns=ALL_ENGINEERED_FEATURES,
    )

    # The training notebook filled occasional engineered NaNs before scaling.
    # A valid >=50-word sample should normally have none. If one remains,
    # use zero only as a defensive fallback after preserving column order.
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = bundle["transformers"]["scaler"]
    tfidf = bundle["transformers"]["tfidf"]

    engineered_scaled = scaler.transform(feature_frame)
    semantic = tfidf.transform([text])

    hybrid = sparse.hstack(
        [semantic, sparse.csr_matrix(engineered_scaled)],
        format="csr",
    )

    return hybrid, engineered


def analyse_text(text: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    words = count_words(text)
    if words < int(bundle["min_words"]):
        raise ValueError(
            f"Text contains {words} words; at least {bundle['min_words']} are required."
        )

    X, engineered = build_hybrid_matrix(text, bundle)
    model = bundle["model"]

    prediction = int(model.predict(X)[0])

    probabilities = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)[0]
        class_to_prob = {
            int(cls): float(prob)
            for cls, prob in zip(model.classes_, raw)
        }
        p_human = class_to_prob.get(0, 0.0)
        p_ai = class_to_prob.get(1, 0.0)
        probabilities = {
            "Human": p_human,
            "AI-generated": p_ai,
        }
        confidence = max(p_human, p_ai)
    else:
        # Defensive fallback. The supplied deployment bundle is calibrated and
        # therefore exposes predict_proba.
        if hasattr(model, "decision_function"):
            decision = float(np.ravel(model.decision_function(X))[0])
            p_ai = 1.0 / (1.0 + np.exp(-decision))
            probabilities = {
                "Human": 1.0 - p_ai,
                "AI-generated": p_ai,
            }
            confidence = max(probabilities.values())
        else:
            probabilities = {
                "Human": 1.0 if prediction == 0 else 0.0,
                "AI-generated": 1.0 if prediction == 1 else 0.0,
            }
            confidence = 1.0

    prediction_name = "AI-generated" if prediction == 1 else "Human-written"

    display_features = {
        name: engineered[name]
        for name, _ in MIN_DISPLAY_FEATURES
    }

    return {
        "prediction": prediction,
        "prediction_name": prediction_name,
        "confidence": float(confidence),
        "probabilities": probabilities,
        "word_count": words,
        "engineered_features": engineered,
        "display_features": display_features,
    }
