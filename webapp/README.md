# AI Tool Survey — Dashboard (Flask)

An interactive dashboard for the survey analysis: overview stats, data exploration
charts, statistical test results, model comparison, an unbiased testing/overfit page,
and a live predictor built on the trained model.

Chart.js is **vendored locally** at `static/js/vendor/chart.umd.js` — no CDN dependency,
no risk of a version mismatch breaking the charts.

## Setup

This app reads data from `../data/processed/`, so it must stay inside the project
folder, as a sibling of `data/` (this is already how it's packaged).

Using the **same virtual environment** as the rest of the project:

```bash
cd ..                          # back to AI_Tool_Survey_Project/, if you're in webapp/
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r webapp/requirements.txt
cd webapp
python app.py
```

## Running it

```bash
python app.py
```

The app automatically finds an open port starting from **5050** (then 5051, 5052,
5060, 5080, 8000, 8080, 8888) — useful since 5000 and 5001 are often already taken
by macOS AirPlay Receiver or other local services. The terminal prints the actual URL:

```
AI Tool Survey Dashboard running at: http://127.0.0.1:5050/
```

Open that URL in your browser. To stop the server, press `Ctrl+C`.

**To force a specific port:** edit the `candidate_ports` list near the bottom of
`app.py` and put your preferred port first.

## Pages

| Route | What it shows |
|---|---|
| `/` | Overview — key stats, top tool, best model score |
| `/explore` | Demographics, tool usage, cross-tabs by age/occupation/residence |
| `/statistics` | Chi-square test results, p-value chart |
| `/models` | Baseline vs tuned model comparison, feature importance |
| `/testing` | Unbiased held-out evaluation — confusion matrix, ROC curves, train-vs-test gap, learning curve (mirrors `10_Testing.ipynb`) |
| `/predict` | Interactive form — predicts most-used tool for a hypothetical respondent |
| `/interpretation` | The written analysis, pulled together from all sections |

## Design

- Light/dark theme toggle (bottom of sidebar), persisted in your browser via
  `localStorage`, and defaults to your OS preference on first visit.
- Fully responsive — the sidebar collapses into a slide-out menu below ~760px.
- All charts are drawn live from the project's own processed CSVs and the saved
  model — nothing is hardcoded, so re-running the notebooks with more survey
  responses updates this dashboard automatically on next page load.

## Notes

- This is a **development server** (Flask's built-in one) — fine for local viewing
  and demos, not for public deployment.
- If you add more respondents and re-run notebooks `02` through `10`, restart this
  app (`Ctrl+C` then `python app.py` again) to pick up the refreshed data and model.
