# AI Club Induction Tasks
 
Solutions to the AI Club induction tasks across two rounds, covering attention mechanism implementation and deep learning-based time-series forecasting.
 
---
 
# 📂 Project Structure
 
```
ai-club-induction-tasks/
│
├── Round2_JOSHUA/
│   ├── Round2_Code_Joshua.py
│   └── Round2_Explanation_Joshua.pdf
│
├── Round3_JOSHUA/
│   ├── Round3_Model_Joshua.ipynb
│   ├── typing_speed_train.csv
│   ├── Round3_Predictions_Joshua.csv
│   └── Round3_Report_Joshua.pdf
```
 
---
 
# 🧠 Round 2 — Memory-Efficient Attention (FlashAttention-style)
 
## Files
 
- `Round2_Code_Joshua.py`
- `Round2_Explanation_Joshua.pdf`
## Description
 
Round 2 implements a **memory-efficient scaled dot-product attention mechanism** in NumPy, closely following the design of FlashAttention 2.0.
 
Rather than materialising the full N×N attention score matrix, the implementation processes queries and keys in **blocks**, computing softmax incrementally using a numerically stable online update. This avoids the quadratic memory cost of naive attention.
 
### Forward Pass
 
```python
def memory_efficient_attention(Q, K, V, block_size=32):
    # Processes Q in outer blocks, K/V in inner blocks
    # Tracks running max (m_i) and normalisation sum (l_i)
    # Updates output (y_i) incrementally without storing full scores
```
 
For each query block `Q_i`:
- Iterates over all key/value blocks `K_j`, `V_j`
- Maintains a running softmax numerator `y_i` and denominator `l_i`
- Updates using the recurrence: `m_new = max(m_i, max_block)`
### Correctness Verification
 
A naive attention baseline is included and used to verify numerical equivalence:
 
```python
def naive_attention(Q, K, V):
    scores = Q @ K.T
    weights = softmax(scores)
    return weights @ V
```
 
The max absolute difference between both implementations is printed at runtime, confirming correctness.
 
### Example
 
```python
n, d = 128, 32
X = np.random.randn(n, d)
out = memory_efficient_attention(X, X, X, block_size=32)
# Output shape: (128, 32)
```
 
The PDF explains the methodology, the backward pass design (activation recomputation rather than storing intermediates), and the connection to FlashAttention 2.0.
 
---
 
# 🤖 Round 3 — Typing Performance Forecasting (LSTM)
 
## Files
 
- `Round3_Model_Joshua.ipynb`
- `typing_speed_train.csv`
- `Round3_Predictions_Joshua.csv`
- `Round3_Report_Joshua.pdf`
## Objective
 
Predict the next typing session's performance metrics from a sequence of historical sessions using an LSTM neural network.
 
---
 
## 📊 Dataset Features
 
| Feature | Description |
|---|---|
| `wpm` | Words per minute |
| `acc` | Typing accuracy (%) |
| `rawWpm` | Raw (uncorrected) typing speed |
| `consistency` | Typing consistency score |
 
---
 
## ⚙️ Workflow
 
### 1. Data Loading & Inspection
 
```python
df = pd.read_csv("typing_speed_train.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())
```
 
### 2. Feature Selection & Normalisation
 
The four numeric features are extracted and normalised using `StandardScaler`:
 
```python
df_numeric = df[['wpm', 'acc', 'rawWpm', 'consistency']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)
```
 
### 3. Sequence Generation
 
Rolling windows of 30 sessions are created. Each window predicts the next session:
 
```python
def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
```
 
Example:
```
Sessions 1–30  → Predict Session 31
Sessions 2–31  → Predict Session 32
```
 
### 4. Train / Validation Split
 
An 80/20 chronological split is used:
 
```python
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
```
 
### 5. Visualisation
 
The notebook includes:
- Per-feature trend plots (Matplotlib)
- Correlation heatmap (Seaborn)
- Dataset statistics and null checks
### 6. LSTM Model
 
```python
model = Sequential([
    LSTM(64, input_shape=(30, 4)),
    Dense(32, activation='relu'),
    Dense(4)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=10, batch_size=32)
```
 
The model predicts all four features simultaneously: `wpm`, `acc`, `rawWpm`, `consistency`.
 
### 7. Prediction Stabilisation
 
The raw model output is blended with the most recent real session to suppress unrealistic jumps:
 
```python
final_pred = 0.65 * pred[0] + 0.35 * last_real
```
 
Additional hard constraints are applied:
 
| Feature | Constraint |
|---|---|
| `wpm` | Capped at `last_real + 5` |
| `rawWpm` | Capped at `last_real + 5` |
| `consistency` | Capped at `last_real + 6` |
| `acc` | Clipped to `[90, 98]` |
 
### 8. Output
 
The stabilised prediction is saved to `Round3_Predictions_Joshua.csv`:
 
```python
pred_df = pd.DataFrame([final_pred], columns=['wpm','acc','rawWpm','consistency'])
pred_df.to_csv("Round3_Predictions_Joshua.csv", index=False)
```
 
---
 
## 🛠 Technologies Used
 
| Category | Libraries |
|---|---|
| Language | Python |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Preprocessing | Scikit-learn (`StandardScaler`) |
| Modelling | TensorFlow / Keras |
 
---
 
## 📦 Installation
 
```bash
git clone https://github.com/YOUR_USERNAME/ai-club-induction-tasks.git
cd ai-club-induction-tasks
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```
 
---
 
## ▶️ Running the Notebook
 
```bash
jupyter notebook Round3_JOSHUA/Round3_Model_Joshua.ipynb
```
 
Run cells sequentially. The final cell writes `Round3_Predictions_Joshua.csv`.
 
---
 
## 📄 Reports
 
PDF reports are included for both rounds:
- `Round2_Explanation_Joshua.pdf` — attention mechanism design, backward pass rationale, FlashAttention comparison
- `Round3_Report_Joshua.pdf` — forecasting methodology, model architecture, prediction logic
---
 
## 🎯 Key Concepts Demonstrated
 
**Round 2**
- Memory-efficient attention (FlashAttention 2.0-style)
- Online softmax with numerical stability
- Block-wise matrix computation
- Correctness verification against naive baseline
**Round 3**
- LSTM-based time-series forecasting
- Feature scaling and sequence generation
- Train/validation splitting
- Prediction stabilisation and constraint clamping
- Multi-output regression
---
 
## 📌 Future Improvements
 
- GRU or Transformer-based forecasting for Round 3
- Hyperparameter tuning (sequence length, LSTM units, epochs)
- Real-time prediction dashboard
- Attention mechanism: full backward pass implementation
- Extended evaluation metrics (MAE, MAPE per feature)
 
