# 🎮 Tic-Tac-Toe AI Arena

> An interactive Streamlit dashboard for analysing the UCI Tic-Tac-Toe Endgame dataset
> with a neon-cyberpunk UI and live board predictor.

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/tictactoe-ai-arena.git
cd tictactoe-ai-arena

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset in the same folder
#    (tic_tac_toe.csv must be in the working directory)

# 4. Launch
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (include `app.py`, `requirements.txt`, `tic_tac_toe.csv`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → **New app**.
3. Select your repo, branch `main`, file `app.py` → **Deploy**.

Done — your app is live in ~2 minutes!

---

## 📦 Project Structure

```
tictactoe-ai-arena/
├── app.py              # Main Streamlit application
├── tic_tac_toe.csv     # UCI dataset (958 endgame states)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ✨ Features

| Tab | What you get |
|-----|-------------|
| **📊 Overview** | Class distribution donut, model comparison bar chart, board heatmap |
| **🤖 Model Analysis** | Confusion matrix, ROC curve, cross-validation box plots, precision/recall/F1 |
| **🎯 Live Predictor** | Interactive 3×3 board — set any endgame state and get an AI prediction with confidence % |
| **📈 Feature Intelligence** | Feature importances, X win-rate by position, symbol distribution by outcome |

**Models included:** Random Forest · Gradient Boosting · Decision Tree · K-Nearest Neighbors

---

## 🗄️ Dataset

**UCI Tic-Tac-Toe Endgame** — 958 board states  
Each of the 9 positions can be `x`, `o`, or `b` (blank).  
Class: `positive` = X wins, `negative` = X does not win.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io) — UI framework  
- [Plotly](https://plotly.com) — interactive charts  
- [scikit-learn](https://scikit-learn.org) — ML models  
- [pandas](https://pandas.pydata.org) / [NumPy](https://numpy.org) — data handling
