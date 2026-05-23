import shap
import joblib

from preprocess import load_and_preprocess
from feature_engineering import create_features

model = joblib.load(
    "saved_models/xgb_model.pkl"
)

df = load_and_preprocess("../data/EV_Predictv_data.csv")

df = create_features(df)

DROP_COLUMNS = [
    "Timestamp",
    "Failure_Probability",
    "RUL",
    "TTF"
]

X = df.drop(columns=DROP_COLUMNS)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)
