import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data = pd.read_csv("AmesHousing.csv")
print("Shape:", data.shape)
data.head()

features = [
    "Gr Liv Area", "Bedroom AbvGr", "Full Bath", "Half Bath", "Garage Cars",
    "Year Built", "Lot Area", "Overall Qual", "Overall Cond", "Total Bsmt SF",
    "Garage Area", "1st Flr SF", "2nd Flr SF", "Fireplaces"
]

X = data[features]
y = data["SalePrice"]

X = X.fillna(0)
y = y.fillna(y.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("🎯 Training Complete with XGBoost")
print("RMSE:", rmse)
print("R² Score:", r2)

joblib.dump(model, "house_price_model.pkl")