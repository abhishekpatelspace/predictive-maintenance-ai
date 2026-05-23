import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier

from preprocess import load_and_preprocess
from feature_engineering import create_features

df = load_and_preprocess("../data/EV_Predictv_data.csv")

df = create_features(df)

df["Failure_Label"] = (
    df["Failure_Probability"] > 0.7
).astype(int)

encoder = LabelEncoder()

df["Maintenance_Type"] = encoder.fit_transform(
    df["Maintenance_Type"]
)

DROP_COLUMNS = [
    "Timestamp",
    "Failure_Probability",
    "Failure_Label",
    "RUL",
    "TTF"
]

X = df.drop(columns=DROP_COLUMNS)
y = df["Failure_Label"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

joblib.dump(model, "saved_models/xgb_model.pkl")
