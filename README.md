# RoboSense: Explainable Failure Prediction for Robotic Arms

Classifies robot assembly failures (collision, obstruction, etc.) from
force/torque sensor time-series, comparing a feature-engineered XGBoost
model against a raw-sequence LSTM, with SHAP explainability.

**Dataset:** UCI Robot Execution Failures (LP1-LP5), force/torque
readings from a pick-and-place robot arm after failure detection.

## Project structure

```
failsafe_project/
├── data/                    # lp1.data.txt ... lp5.data.txt (raw dataset)
├── data_loader.py           # parses the raw LP-format files
├── features.py               # statistical + FFT feature extraction
├── train.py                  # trains XGBoost + LSTM, runs SHAP, saves results_summary.json
├── make_plots.py             # generates confusion matrix / SHAP / comparison plots for slides
├── app.py                    # Streamlit live demo
├── requirements.txt
└── results_summary.json      # generated after running train.py
```

## Step-by-step: reproduce from scratch

### 1. Set up environment
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset
Download the 5 LP files into `data/`:
```bash
mkdir -p data
for f in lp1 lp2 lp3 lp4 lp5; do
  curl -o data/${f}.data.txt \
    https://raw.githubusercontent.com/MaxBenChrist/robot-failure-dataset/master/${f}.data.txt
done
```
(Mirror of the official UCI Robot Execution Failures dataset.)

### 3. Train and evaluate
```bash
python3 train.py --lp data/lp1.data.txt
```
This will:
- Parse the raw sensor traces
- Extract 48 statistical/FFT features per instance
- Apply SMOTE to the training split only (never the test split)
- Train XGBoost on features and an LSTM on raw sequences
- Print classification reports and macro-F1 for both
- Run SHAP on the XGBoost model and print the top driving features
- Save everything to `results_summary.json`

Try the other sub-datasets too — each is a different failure scenario:
```bash
python3 train.py --lp data/lp2.data.txt   # transfer failures
python3 train.py --lp data/lp5.data.txt   # motion-with-part failures (largest, 164 instances)
```

### 4. Generate slide-ready plots
```bash
python3 make_plots.py
```
Produces `plot_xgb_confusion.png`, `plot_lstm_confusion.png`,
`plot_shap_importance.png`, `plot_model_comparison.png` — drop these
straight into your presentation.

### 5. Run the live demo (for your viva)
```bash
streamlit run app.py
```
Opens a browser app where you pick a sample sensor trace, see the
predicted failure type, its confidence, and the SHAP explanation for
why the model predicted it. This is what you demo live to the judges.

## Results (LP1, 88 instances, 4 classes)

| Model | Macro F1 |
|---|---|
| XGBoost (48 engineered features) | ~0.96 |
| LSTM (raw 15-step sequence) | ~0.54 |

**Why XGBoost wins here:** the dataset is small (88 instances) and
failure signatures show up clearly in simple statistics (peak force,
variance, dominant vibration frequency) rather than needing the model
to learn long-range temporal dependencies. LSTMs typically need far
more data to outperform well-engineered features — a genuinely useful,
honest finding to report rather than a limitation to hide.

## Report / slide structure (5-6 slides)

1. Problem: predictive maintenance for robotic assembly systems
2. Architecture (data → features → model comparison → SHAP)
3. XGBoost vs LSTM results + why one wins
4. SHAP explainability — which sensor readings drive each failure type
5. Live demo screenshot
6. Limitations (small dataset, single robot arm) + next steps (more
   data, sensor fusion, deployment on real hardware)

## Submitting to GitHub

```bash
cd robosense_project
git init
git add .
git commit -m "RoboSense: robot failure diagnosis with XGBoost/LSTM comparison + SHAP"
git branch -M main
git remote add origin https://github.com/<your-username>/robosense.git
git push -u origin main
```
Add a `.gitignore` excluding `venv/` and `__pycache__/` before committing.
