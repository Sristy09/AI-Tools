"""
AI Tool Usage & Adoption Survey — Analysis Dashboard
Flask application serving the survey analysis, statistics, model results,
and an interactive prediction tool built on the trained Random Forest model.
"""
import os
import socket
import warnings
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

warnings.filterwarnings('ignore')

app = Flask(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
# The webapp lives inside the project as a sibling of data/, e.g.:
#   AI_Tool_Survey_Project/data/processed/...
#   AI_Tool_Survey_Project/webapp/app.py   <- this file
DATA_DIR = os.path.join(HERE, '..', 'data', 'processed')

# ---------------------------------------------------------------------------
# Load data once at startup
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA_DIR, 'AI_Tool_Survey_Cleaned.csv'))
users = df[df['Is_AI_User']].copy()

stat_summary = pd.read_csv(os.path.join(DATA_DIR, 'Statistical_Test_Summary.csv'))
model_results = pd.read_csv(os.path.join(DATA_DIR, 'Model_Comparison_Results.csv'), index_col=0)
feature_table = pd.read_csv(os.path.join(DATA_DIR, 'features', 'Engineered_Features.csv'))

model_bundle = joblib.load(os.path.join(DATA_DIR, 'models', 'best_model.pkl'))
MODEL = model_bundle['model']
MODEL_NAME = model_bundle['model_name']
LABEL_ENCODER = model_bundle['label_encoder']
FEATURE_COLUMNS = model_bundle['feature_columns']
CV_MACRO_F1 = model_bundle['cv_macro_f1']
BEST_PARAMS = model_bundle.get('best_params', {})

ORDER_AGE = ['13 - 17', '18 - 24', '25 - 34', '35 - 44', '45 and above']
OCCUPATIONS = sorted(feature_table['Occupation'].unique().tolist())
RESIDENCE_TYPES = ['Rural', 'Semi - Urban', 'Urban']

# Feature importances (works for tree models; falls back to |coef| for linear models)
if hasattr(MODEL, 'feature_importances_'):
    _importances = pd.Series(MODEL.feature_importances_, index=FEATURE_COLUMNS)
elif hasattr(MODEL, 'coef_'):
    _importances = pd.Series(np.abs(MODEL.coef_).mean(axis=0), index=FEATURE_COLUMNS)
else:
    _importances = pd.Series(dtype=float)
FEATURE_IMPORTANCE = _importances.sort_values(ascending=False)


def count_items(cell):
    if pd.isna(cell):
        return 0
    return len([x for x in str(cell).split(';') if x.strip()])


# ---------------------------------------------------------------------------
# Unbiased test-set evaluation (mirrors 10_Testing.ipynb)
# The MODEL above was refit on ALL labeled rows for best prediction quality,
# so it isn't valid for a "how well does this generalize" claim. Here we
# rebuild the same architecture with the same tuned hyperparameters, fit it
# on X_train only, and evaluate on X_test only -- data it has never seen.
# Computed once at startup so page loads stay fast.
# ---------------------------------------------------------------------------
TESTING = None
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, label_binarize
    from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                                  f1_score, roc_curve, auc)
    import xgboost as xgb

    ml_dir = os.path.join(DATA_DIR, 'ml_ready')
    Xtr = pd.read_csv(os.path.join(ml_dir, 'X_train.csv'))
    Xte = pd.read_csv(os.path.join(ml_dir, 'X_test.csv'))
    ytr = pd.read_csv(os.path.join(ml_dir, 'y_train.csv')).iloc[:, 0]
    yte = pd.read_csv(os.path.join(ml_dir, 'y_test.csv')).iloc[:, 0]

    builders = {
        'Logistic Regression': lambda p: LogisticRegression(max_iter=1000, class_weight='balanced', **p),
        'Decision Tree': lambda p: DecisionTreeClassifier(random_state=42, class_weight='balanced', **p),
        'Random Forest': lambda p: RandomForestClassifier(random_state=42, class_weight='balanced', **p),
        'XGBoost': lambda p: xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, **p),
    }
    test_le = LabelEncoder()
    ytr_enc = test_le.fit_transform(ytr)
    yte_enc = test_le.transform(yte)

    test_model = builders[MODEL_NAME](BEST_PARAMS)
    if MODEL_NAME == 'XGBoost':
        test_model.fit(Xtr, ytr_enc)
        pred_train = test_le.inverse_transform(test_model.predict(Xtr))
        pred_test = test_le.inverse_transform(test_model.predict(Xte))
        proba_test = test_model.predict_proba(Xte)
        test_classes = list(test_le.classes_)
    else:
        test_model.fit(Xtr, ytr)
        pred_train = test_model.predict(Xtr)
        pred_test = test_model.predict(Xte)
        proba_test = test_model.predict_proba(Xte)
        test_classes = list(test_model.classes_)

    labels_sorted = sorted(yte.unique())
    cm = confusion_matrix(yte, pred_test, labels=labels_sorted)
    report = classification_report(yte, pred_test, zero_division=0, output_dict=True)

    yte_bin = label_binarize(yte, classes=test_classes)
    roc_data = []
    for i, cls in enumerate(test_classes):
        fpr, tpr, _ = roc_curve(yte_bin[:, i], proba_test[:, i])
        step = max(1, len(fpr) // 30)
        roc_data.append({
            'tool': cls,
            'auc': round(float(auc(fpr, tpr)), 3),
            'fpr': [round(float(x), 3) for x in fpr[::step]],
            'tpr': [round(float(x), 3) for x in tpr[::step]],
        })

    TESTING = {
        'labels': labels_sorted,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'train_accuracy': round(float(accuracy_score(ytr, pred_train)), 3),
        'test_accuracy': round(float(accuracy_score(yte, pred_test)), 3),
        'train_f1': round(float(f1_score(ytr, pred_train, average='macro', zero_division=0)), 3),
        'test_f1': round(float(f1_score(yte, pred_test, average='macro', zero_division=0)), 3),
        'roc': roc_data,
        'n_train': len(ytr),
        'n_test': len(yte),
    }
except Exception as e:  # pragma: no cover - defensive; testing page shows a friendly message instead
    print(f"[warning] Could not compute testing evaluation: {e}")
    TESTING = None


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html', active='home')


@app.route('/explore')
def explore():
    return render_template('explore.html', active='explore')


@app.route('/statistics')
def statistics():
    return render_template('statistics.html', active='statistics')


@app.route('/models')
def models():
    return render_template('models.html', active='models')


@app.route('/testing')
def testing():
    return render_template('testing.html', active='testing', available=TESTING is not None,
                            model_name=MODEL_NAME)


@app.route('/predict')
def predict_page():
    return render_template(
        'predict.html',
        active='predict',
        occupations=OCCUPATIONS,
        residence_types=RESIDENCE_TYPES,
        model_name=MODEL_NAME,
        cv_f1=round(CV_MACRO_F1, 3),
    )


@app.route('/interpretation')
def interpretation():
    return render_template('interpretation.html', active='interpretation')


# ---------------------------------------------------------------------------
# API: overview stats (used by the home page hero)
# ---------------------------------------------------------------------------
@app.route('/api/overview')
def api_overview():
    total = len(df)
    ai_users = int(df['Is_AI_User'].sum())
    return jsonify({
        'total_respondents': total,
        'ai_users': ai_users,
        'ai_user_pct': round(ai_users / total * 100, 1),
        'best_model': MODEL_NAME,
        'cv_macro_f1': round(CV_MACRO_F1, 3),
        'significant_tests': int(stat_summary['Significant (p<0.05)'].sum()),
        'total_tests': len(stat_summary),
        'top_tool': users['Most_Used_Tool'].value_counts().idxmax(),
    })


# ---------------------------------------------------------------------------
# API: EDA / demographics
# ---------------------------------------------------------------------------
@app.route('/api/demographics')
def api_demographics():
    def counts(col, order=None):
        vc = df[col].value_counts(dropna=True)
        if order:
            vc = vc.reindex([o for o in order if o in vc.index])
        return {'labels': vc.index.tolist(), 'values': [int(v) for v in vc.values]}

    return jsonify({
        'age': counts('Age_Group', ORDER_AGE),
        'gender': counts('Gender'),
        'occupation': counts('Occupation'),
        'residence': counts('Residence_Type'),
        'uses_ai': counts('Uses_AI_Tools'),
    })


@app.route('/api/tool-usage')
def api_tool_usage():
    tool_counts = users['Most_Used_Tool'].value_counts()
    known = users['AI_Tools_Known'].dropna().str.split('; ').explode().str.strip()
    known_counts = known.value_counts()

    purpose = users['Usage_Purpose'].dropna().str.split('; ').explode().str.strip()
    purpose_counts = purpose.value_counts().head(8)

    freq_order = ['Once a month', 'Several times a month', 'Once a week',
                  'Several times a week', 'Once a day', 'Several times a day']
    freq = users['Usage_Frequency'].value_counts()
    freq = freq.reindex([f for f in freq_order if f in freq.index])

    sat_order = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
    sat = users['Satisfaction'].value_counts()
    sat = sat.reindex([s for s in sat_order if s in sat.index])

    return jsonify({
        'most_used_tool': {'labels': tool_counts.index.tolist(), 'values': [int(v) for v in tool_counts.values]},
        'tools_known': {'labels': known_counts.index.tolist(), 'values': [int(v) for v in known_counts.values]},
        'usage_purpose': {'labels': purpose_counts.index.tolist(), 'values': [int(v) for v in purpose_counts.values]},
        'usage_frequency': {'labels': freq.index.tolist(), 'values': [int(v) for v in freq.values]},
        'satisfaction': {'labels': sat.index.tolist(), 'values': [int(v) for v in sat.values]},
    })


@app.route('/api/cross-tab/<dimension>')
def api_cross_tab(dimension):
    dim_map = {'age': 'Age_Group', 'occupation': 'Occupation', 'residence': 'Residence_Type', 'gender': 'Gender'}
    if dimension not in dim_map:
        return jsonify({'error': 'unknown dimension'}), 404

    col = dim_map[dimension]
    ct = pd.crosstab(users[col], users['Most_Used_Tool'])
    # keep only the top tool columns to keep the chart legible
    top_tools = users['Most_Used_Tool'].value_counts().head(4).index.tolist()
    ct = ct[[c for c in top_tools if c in ct.columns]]

    if dimension == 'age':
        ct = ct.reindex([a for a in ORDER_AGE if a in ct.index])
    else:
        ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    return jsonify({
        'labels': ct.index.tolist(),
        'series': [{'name': tool, 'values': [int(v) for v in ct[tool].values]} for tool in ct.columns],
    })


# ---------------------------------------------------------------------------
# API: statistics
# ---------------------------------------------------------------------------
@app.route('/api/statistics')
def api_statistics():
    records = stat_summary.to_dict(orient='records')
    for r in records:
        r['Chi2'] = round(r['Chi2'], 2)
        r['p_value'] = round(r['p_value'], 4)
    return jsonify({'tests': records})


# ---------------------------------------------------------------------------
# API: model comparison
# ---------------------------------------------------------------------------
@app.route('/api/model-results')
def api_model_results():
    records = []
    for idx, row in model_results.iterrows():
        records.append({
            'label': idx,
            'accuracy': round(row['Accuracy'], 3),
            'precision': round(row['Precision (macro)'], 3),
            'recall': round(row['Recall (macro)'], 3),
            'f1': round(row['F1 (macro)'], 3),
        })
    return jsonify({'results': records, 'best_model': MODEL_NAME, 'best_cv_f1': round(CV_MACRO_F1, 3)})


@app.route('/api/feature-importance')
def api_feature_importance():
    top = FEATURE_IMPORTANCE.head(12)
    return jsonify({
        'labels': top.index.tolist(),
        'values': [round(float(v), 4) for v in top.values],
    })


@app.route('/api/testing')
def api_testing():
    if TESTING is None:
        return jsonify({'available': False}), 200
    return jsonify({'available': True, **TESTING})


# ---------------------------------------------------------------------------
# API: prediction
# ---------------------------------------------------------------------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(force=True)

    row = {
        'Age_Group_Code': int(payload.get('age_group_code', 2)),
        'Gender': payload.get('gender', 'Female'),
        'Occupation': payload.get('occupation', 'Student'),
        'Residence_Type': payload.get('residence_type', 'Urban'),
        'Usage_Frequency_Code': int(payload.get('usage_frequency_code', 4)),
        'Satisfaction_Code': int(payload.get('satisfaction_code', 3)),
        'Num_Tools_Known': int(payload.get('num_tools_known', 2)),
        'Num_Usage_Purposes': int(payload.get('num_usage_purposes', 3)),
        'Num_Usage_Contexts': int(payload.get('num_usage_contexts', 2)),
        'Num_Concerns': int(payload.get('num_concerns', 2)),
    }
    row['Is_Heavy_User'] = int(row['Usage_Frequency_Code'] >= 5)
    row['Has_Multiple_Concerns'] = int(row['Num_Concerns'] > 1)
    row['Knows_Multiple_Tools'] = int(row['Num_Tools_Known'] > 1)

    input_df = pd.DataFrame([row])
    encoded = pd.get_dummies(input_df, columns=['Gender', 'Occupation', 'Residence_Type'])
    encoded = encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    if MODEL_NAME == 'XGBoost':
        pred_enc = MODEL.predict(encoded)
        prediction = LABEL_ENCODER.inverse_transform(pred_enc)[0]
        proba = MODEL.predict_proba(encoded)[0]
        classes = LABEL_ENCODER.inverse_transform(np.arange(len(proba)))
    else:
        prediction = MODEL.predict(encoded)[0]
        proba = MODEL.predict_proba(encoded)[0]
        classes = MODEL.classes_

    probabilities = sorted(
        [{'tool': str(c), 'probability': round(float(p), 3)} for c, p in zip(classes, proba)],
        key=lambda x: -x['probability']
    )

    return jsonify({
        'prediction': str(prediction),
        'probabilities': probabilities,
        'model_used': MODEL_NAME,
    })


# ---------------------------------------------------------------------------
# Port handling — auto-fallback if the requested port is busy
# ---------------------------------------------------------------------------
def find_open_port(candidates):
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return None


if __name__ == '__main__':
    candidate_ports = [5050, 5051, 5052, 5060, 5080, 8000, 8080, 8888]
    port = find_open_port(candidate_ports)
    if port is None:
        raise SystemExit(
            f"All candidate ports are busy: {candidate_ports}. "
            f"Edit `candidate_ports` in app.py to try others."
        )
    print(f"\n  AI Tool Survey Dashboard running at: http://127.0.0.1:{port}/\n")
    # use_reloader=False: the auto-reloader re-executes this file in a subprocess,
    # which would run find_open_port() a second time and could pick a different
    # port than the one just printed above. Restart the script manually after
    # editing files instead.
    app.run(debug=True, port=port, host='127.0.0.1', use_reloader=False)
