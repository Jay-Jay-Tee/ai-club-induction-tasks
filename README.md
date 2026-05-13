# AI Club Induction Tasks

A collection of technical induction tasks completed for the AI Club selection process, covering:

- Algorithmic problem solving
- Data analysis
- Time-series forecasting
- LSTM neural networks
- Machine learning workflows
- Visualization and prediction systems

---

# 📂 Project Structure

```txt
ai-club-induction-tasks/
│
├── Round2_JOSHUA/
│   ├── Round2_Code_Joshua.py
│   └── Round2_Explanation_Joshua.pdf
│
├── Round3_JOSHUA/
│   ├── Round3_Model_Joshua.ipynb
│   ├── Round3_Predictions_Joshua.csv
│   ├── Round3_Report_Joshua.pdf
│   └── typing_speed_train.csv
```

---

# 🚀 Overview

This repository contains solutions and implementations developed during the AI Club induction process.

The tasks focused on:
- analytical thinking
- implementation skills
- machine learning understanding
- data preprocessing
- forecasting models
- explainability

---

# 🧠 Round 2 — Problem Solving

## Files

- `Round2_Code_Joshua.py`
- `Round2_Explanation_Joshua.pdf`

## Description

Round 2 focuses on algorithmic and logical problem solving using Python.

The implementation emphasizes:
- efficient logic
- clean structure
- correctness
- explanatory reasoning

The accompanying PDF explains:
- methodology
- thought process
- optimizations
- final approach

---

# 🤖 Round 3 — Typing Performance Forecasting

## Files

- `Round3_Model_Joshua.ipynb`
- `typing_speed_train.csv`
- `Round3_Predictions_Joshua.csv`
- `Round3_Report_Joshua.pdf`

---

# 📊 Objective

The goal of Round 3 is to predict future typing performance metrics using historical typing data and deep learning.

The project uses an LSTM neural network to model sequential typing behavior.

---

# 📈 Dataset Features

The dataset contains:

| Feature | Description |
|---|---|
| `wpm` | Words per minute |
| `acc` | Typing accuracy |
| `rawWpm` | Raw typing speed |
| `consistency` | Typing consistency |

---

# ⚙️ Workflow

## 1. Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv("typing_speed_train.csv")
```

---

## 2. Data Preprocessing

Relevant numerical features are selected and normalized using `StandardScaler`.

This ensures stable neural network training.

---

## 3. Sequence Generation

The data is converted into sequential windows of 30 entries.

Each sequence predicts the next typing session.

Example:

```txt
Sessions 1–30 → Predict Session 31
Sessions 2–31 → Predict Session 32
```

---

## 4. Visualization

The notebook includes:
- feature trend plots
- correlation heatmaps
- dataset statistics
- null-value inspection

Used libraries:
- Matplotlib
- Seaborn

---

## 5. LSTM Model

The forecasting model uses:

```python
LSTM → Dense(ReLU) → Dense(Output)
```

Architecture:
- LSTM(64)
- Dense(32, relu)
- Dense(4)

The model predicts:
- future WPM
- future accuracy
- future rawWpm
- future consistency

---

# 🔥 Prediction Stabilization

The final prediction blends:
- 65% model prediction
- 35% latest real typing record

```python
final_pred = 0.65 * pred[0] + 0.35 * last_real
```

This reduces unrealistic fluctuations and stabilizes forecasting.

Additional constraints are applied to:
- prevent impossible jumps
- maintain realistic accuracy ranges

---

# 🛠 Technologies Used

## Languages
- Python

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai-club-induction-tasks.git
cd ai-club-induction-tasks
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

---

# ▶️ Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```txt
Round3_JOSHUA/Round3_Model_Joshua.ipynb
```

Run cells sequentially.

---

# 📄 Reports

The repository includes PDF reports explaining:
- methodology
- reasoning
- implementation details
- prediction logic

---

# 🎯 Key Concepts Demonstrated

- Time-series forecasting
- LSTM neural networks
- Data preprocessing
- Feature scaling
- Sequential learning
- Visualization
- Prediction stabilization
- Machine learning workflows

---

# 📌 Future Improvements

Possible future enhancements:
- GRU/Transformer-based forecasting
- Hyperparameter tuning
- Better sequence optimization
- Real-time prediction dashboards
- Advanced evaluation metrics

---
