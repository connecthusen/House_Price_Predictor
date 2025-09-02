# 🏡 House Price Prediction using XGBoost

## 1️⃣ Project Overview
This project predicts **house prices** based on multiple property features using **XGBoost regression**. Users can input house details and get an **estimated price** in real-time. The model is trained on the **Ames Housing dataset**, which contains detailed information about residential homes.

**Goal:** Build a predictive model that accurately estimates the selling price of a house using relevant property features.  

---

## 2️⃣ Dataset

- **Source:** [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  
- **Number of records:** 2,930+ rows (houses)  
- **Features:** 14 features selected based on correlation with house price  

### Features Used

| Feature | Description |
|---------|-------------|
| `Gr Liv Area` | Above ground living area (in sq ft) |
| `Bedroom AbvGr` | Number of bedrooms above ground |
| `Full Bath` | Number of full bathrooms |
| `Half Bath` | Number of half bathrooms |
| `Garage Cars` | Garage capacity (number of cars) |
| `Year Built` | Year the house was built |
| `Lot Area` | Total lot size (in sq ft) |
| `Overall Qual` | Overall material and finish quality (1–10) |
| `Overall Cond` | Overall condition rating (1–10) |
| `Total Bsmt SF` | Total basement area (in sq ft) |
| `Garage Area` | Garage area (in sq ft) |
| `1st Flr SF` | First floor area (in sq ft) |
| `2nd Flr SF` | Second floor area (in sq ft) |
| `Fireplaces` | Number of fireplaces |

> These features were chosen because they strongly influence house prices in the dataset.

---

## 3️⃣ Technology Stack

| Category | Tools / Libraries | Purpose |
|----------|-----------------|---------|
| Programming | Python | Core language for data processing, training, and app |
| Data Handling | Pandas, NumPy | Data manipulation and numerical computations |
| Machine Learning | scikit-learn | Train-test split, scaling, evaluation metrics |
| ML Model | XGBoost | Gradient boosting regression model for price prediction |
| Model Persistence | Joblib | Save/load trained model and scaler |
| UI / Deployment | Gradio | Web interface for interactive predictions |

---

## 4️⃣ Project Structure

house_price_prediction/
│
├── AmesHousing.csv # Dataset
├── train_model.py # Script to train the XGBoost model
├── house_price_model.pkl # Trained XGBoost model (saved)
├── scaler.pkl # Saved StandardScaler
├── app.py # Gradio app to predict house prices
└── README.md # Project documentation

---

## 5️⃣ Data Preprocessing

- **Missing Values:**  
  - Replaced missing numeric values in features with `0`.  
  - Replaced missing target values (`SalePrice`) with the **median** of the column.
- **Scaling:**  
  - Used **StandardScaler** to scale feature values so that the model converges faster and predictions are more accurate.

---

## 6️⃣ Model Training

- **Model Used:** `XGBRegressor` from XGBoost library  
- **Hyperparameters:**
  - `n_estimators=500` – Number of trees
  - `learning_rate=0.05` – Step size shrinkage
  - `max_depth=6` – Maximum depth of a tree
  - `random_state=42` – For reproducibility
  - `n_jobs=-1` – Use all CPU cores  

**Training Steps:**
1. Split dataset into **train** (80%) and **test** (20%).  
2. Scale features using `StandardScaler`.  
3. Train the XGBoost model on scaled training data.  
4. Evaluate on test data using **RMSE** and **R² Score**.  

**Evaluation Metrics:**
- **RMSE (Root Mean Squared Error):** Measures prediction error in USD.  
- **R² Score:** Measures the proportion of variance explained by the model.  

Sample output after training:

🎯 Training Complete
RMSE: 24567.89
R² Score: 0.92
✅ Model and scaler saved!


> High R² indicates the model fits the data well.

---

## 7️⃣ Model & Scaler

- **house_price_model.pkl** – Trained XGBoost model  
- **scaler.pkl** – StandardScaler to scale user inputs before prediction  

> Both files are required for running the interactive prediction app.

---

## 8️⃣ Gradio Web App

The app allows users to input property features and predicts the house price.

**Steps to Use:**

1. Ensure `house_price_model.pkl` and `scaler.pkl` are in the same folder.  
2. Run the app:

```bash
python app.py
Open the local Gradio URL in your browser.

Enter house details and click Predict Price.

Get the estimated price instantly.

App Inputs:

Living Area (sq ft)

Bedrooms

Full Bathrooms

Half Bathrooms

Garage Capacity (cars)

Year Built

Lot Area (sq ft)

Overall Quality Rating (1–10)

Overall Condition Rating (1–10)

Basement Area (sq ft)

Garage Area (sq ft)

First Floor Area (sq ft)

Second Floor Area (sq ft)

Number of Fireplaces

Output: Predicted house price in USD.
