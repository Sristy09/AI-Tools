# Analysis and Prediction of AI Tool Preferences, Usage Patterns, and Adoption Across User Groups

Internship project — Shristi Khati, BIT 8th Semester, Gandaki University

## Folder structure

```
AI_Tool_Survey_Project/
├── README.md
├── requirements.txt
├── 01_Data_Inspection.ipynb        # shape, dtypes, missing, duplicates, categories (raw data only)
├── 02_Data_Cleaning.ipynb          # standardize, fix bugs, skip-logic -> saves cleaned CSV
├── 03_EDA.ipynb                    # who uses AI, tools used, age/occupation/residence patterns
├── 04_Statistical_Analysis.ipynb   # chi-square tests, Spearman correlation, p-value diagram
├── 05_Feature_Engineering.ipynb    # engineered count features from multi-select columns
├── 06_Model_Preparation.ipynb      # encode, train/test split, class imbalance handling
├── 07_Model_Comparison.ipynb       # 4 models baseline + Optuna-tuned, confusion matrices,
│                                    #   train-vs-test diagrams, saves the winning model
├── 08_Prediction.ipynb             # load best model, predict on data + a new hypothetical case
├── 09_Interpretation.ipynb         # feature importance + written interpretation
├── 10_Testing.ipynb                # unbiased test-set evaluation: confusion matrix, ROC curves,
│                                    #   classification report, learning curve (train vs CV score)
├── webapp/                         # interactive Flask dashboard (see webapp/README.md)
│   ├── app.py
│   ├── requirements.txt            # just Flask — install into the same venv
│   ├── templates/
│   └── static/
└── data/
    ├── raw/                        # original Google Form export (never edit this file)
    │   └── AI_Tool_Usage_and_Adoption_Survey__Responses_.xlsx
    └── processed/                  # created automatically as you run the notebooks
        ├── AI_Tool_Survey_Cleaned.csv / .xlsx
        ├── features/Engineered_Features.csv
        ├── ml_ready/                 # X_train, X_test, y_train, y_test, sample weights
        ├── models/best_model.pkl     # winning model + its tuned hyperparameters
        ├── Statistical_Test_Summary.csv
        ├── Model_Comparison_Results.csv
        ├── Predictions.csv
        └── Testing_Report.csv        # classification report from the unbiased test evaluation
```

## Notebook style

Each notebook is written as a sequence of **code cells with comment headers**
(`# --- Section name ---`) rather than separate markdown cells between every step —
keeps the notebook compact and lets you read the logic and its explanation in one place.

## Setup (do this once)

Open a terminal **inside this project folder** and run:

**Windows (PowerShell):**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the notebooks

With the virtual environment active:
```
jupyter notebook
```

Run them **in order, top to bottom** — each depends on the previous one's output in `data/processed/`:

1. **01_Data_Inspection** — understand the raw data before touching it
2. **02_Data_Cleaning** — the only notebook that writes the cleaned dataset
3. **03_EDA** — visual answers to the core research questions
4. **04_Statistical_Analysis** — chi-square tests for group differences
5. **05_Feature_Engineering** — turns cleaned data into a model-ready feature table
6. **06_Model_Preparation** — encodes categoricals, splits train/test, computes class weights
7. **07_Model_Comparison** — trains 4 models baseline + Optuna-tuned, saves the best one
8. **08_Prediction** — loads the saved model and generates predictions
9. **09_Interpretation** — feature importance + a template for your written conclusions
10. **10_Testing** — retrains the winning architecture on train-only data for an honest,
    unseen-data evaluation (ROC curves, learning curve, train-vs-test comparison)

## Interactive dashboard

After running notebooks `01` through `07` at least once (so `data/processed/` is
populated), you can browse everything in a live web dashboard instead of the notebooks:

```bash
cd webapp
pip install flask
python app.py
```

Six pages: Overview, Data Exploration, Statistical Tests, Model Comparison, **Testing &
Overfit Check** (mirrors notebook 10 — confusion matrix, ROC curves, train-vs-test gap),
and Try the Predictor. See `webapp/README.md` for details, including the automatic port
fallback (useful since 5000/5001 are often already taken locally) and the light/dark
theme toggle.

## Key design decisions (for your methodology section)

- **PII removed:** `Email`, `Home_Address` dropped; a phone number typed into the comments
  field is redacted.
- **A subtle bug fixed:** the checkbox option *"Creating images, videos, or other media"*
  contains its own commas, which corrupts naive comma-splitting of multi-select answers for
  72 respondents unless handled explicitly (done in `02_Data_Cleaning`, with a before/after
  diagram showing the fix).
- **Columns excluded from ML** (`05` onward): `Respondent_ID`, `Timestamp`, `Comments`,
  `Feedback_Theme` — none carry predictive signal for tool preference.
- **Class imbalance** is handled with balanced sample weights (not row duplication or
  deletion).
- **Hyperparameter optimization:** Optuna tunes all four models by maximizing macro-F1
  under repeated stratified cross-validation on the training set.
- **Why three different evaluations exist across 07, 08, and 10:** `07` reports both a
  single holdout test (small, ~47 rows) and 5-fold CV across all 186 rows (used to pick
  the winner); `08` shows in-sample predictions on the model refit on all data (for making
  new predictions, not a performance claim); `10` retrains the winning architecture on
  train-only data for the one genuinely unbiased "how well does this generalize" answer,
  including ROC curves and a learning curve to visualize the overfitting directly.

## Notes

- `venv/` should not be shared or committed. If using git, ignore `venv/` and
  `data/processed/`.
- Re-running `01`/`02` is always safe — they start fresh from the untouched raw file.
- Sample size is currently ~200 responses (186 usable for modeling); model performance is
  moderate as a result — an honest, reportable limitation (see `10_Testing.ipynb`'s
  learning curve), not a bug, and a case for continuing data collection.
