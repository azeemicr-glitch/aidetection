# Hybrid AI-Generated Text Detection Framework

A Streamlit demonstration interface for the MSc research artefact:

**Development and Evaluation of a Hybrid Machine Learning Framework for Detecting AI-Generated Text Using Statistical, Stylometric and Semantic Features**

## What this repository contains

- `app.py` — Streamlit user interface
- `utils.py` — inference preprocessing and feature extraction
- `deployment_best_hybrid.joblib` — trained Hybrid Linear SVM deployment bundle
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — simple Streamlit theme/configuration
- `tests/test_pipeline.py` — automated technical smoke tests
- `FUNCTIONAL_TEST_PLAN.csv` — functional test plan for artefact validation

The training datasets are intentionally **not** included. The deployed app performs inference only and does not retrain the model.

## Model

The supplied deployment bundle contains:

- calibrated Linear SVM
- fitted TF-IDF vectoriser
- fitted StandardScaler
- statistical features
- stylometric features
- Hybrid feature definition
- model-validation metadata

The model uses a **minimum recommended input length of 50 words**.

## Important limitation

This is a research prototype. The dissertation experiments found strong in-domain/combined held-out performance, but substantial performance deterioration during cross-dataset transfer:

- HC3 → DAIGT F1: **54.68%**
- DAIGT → HC3 F1: **65.53%**

A prediction from this app must therefore **not** be treated as definitive evidence of AI authorship.

## Local run

Python 3.11 is recommended because the supplied model bundle was serialised with `scikit-learn 1.6.1`.

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run automated tests:

```bash
python -m unittest discover -s tests -v
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository, for example `hybrid-ai-text-detector`.
2. Upload the complete contents of this project folder to the repository root.
3. Open Streamlit Community Cloud and sign in with GitHub.
4. Choose **Create app**.
5. Select the repository and `main` branch.
6. Set the entrypoint file to `app.py`.
7. In advanced settings, select **Python 3.11**.
8. Deploy.
9. Once the app is live, repeat the functional test plan against the hosted version.

`requirements.txt` pins `scikit-learn==1.6.1` because the deployment bundle was created with that version.

## Recommended dissertation evidence after deployment

Capture only a small number of screenshots:

1. clean landing/input state;
2. successful valid-text prediction with confidence and feature summary;
3. short-text validation warning;
4. optional technical/model information panel.

The functional test outcomes can then be added to the Chapter 4 testing/validation section.
