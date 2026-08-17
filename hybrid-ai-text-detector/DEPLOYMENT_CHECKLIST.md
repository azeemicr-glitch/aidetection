# Deployment Checklist

## GitHub

1. Unzip `hybrid-ai-text-detector-github-ready.zip`.
2. Create a GitHub repository named, for example, `hybrid-ai-text-detector`.
3. Upload the **contents inside** the `hybrid-ai-text-detector` folder to the repository root.
4. Confirm that these files are visible at the repository root:
   - `app.py`
   - `utils.py`
   - `requirements.txt`
   - `deployment_best_hybrid.joblib`
   - `README.md`
   - `FUNCTIONAL_TEST_PLAN.csv`
   - `MODEL_CARD.json`
   - `.streamlit/config.toml`
   - `tests/test_pipeline.py`

## Streamlit Community Cloud

1. Sign in with GitHub.
2. Create a new app from the GitHub repository.
3. Branch: `main`
4. Entrypoint: `app.py`
5. In advanced settings, choose **Python 3.11**.
6. Deploy.
7. Open the generated `streamlit.app` URL.
8. Run the functional tests in `FUNCTIONAL_TEST_PLAN.csv`.
9. Fill the `Actual result` and `Status` columns after testing the hosted app.

## What not to upload

The HC3 and DAIGT raw/training datasets are not needed for the deployed application.
The app performs inference only using the saved `.joblib` bundle.

## If deployment fails

Check the Streamlit deployment logs first.

The most important compatibility requirement is:

`scikit-learn==1.6.1`

The supplied model bundle was serialized with that scikit-learn version.

## Dissertation evidence to capture

After deployment, retain:
- one screenshot of the landing/input page;
- one screenshot of a successful prediction;
- one screenshot of the short-input warning;
- completed functional test-plan table;
- public Streamlit URL;
- GitHub repository URL.

These can be incorporated into the final artefact implementation/testing section.
