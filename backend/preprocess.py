import pandas as pd

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.sort_values("Timestamp")

    df = df.drop_duplicates()

    df = df.ffill()

    return df
