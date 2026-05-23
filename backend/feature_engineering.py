def create_features(df):

    df["Battery_Degradation"] = 100 - df["SoH"]

    df["Motor_Stress"] = (
        df["Motor_Temperature"]
        * df["Motor_Vibration"]
    )

    df["Usage_Intensity"] = (
        df["Driving_Speed"]
        * df["Load_Weight"]
    )

    df["Brake_Health"] = (
        100 - df["Brake_Pad_Wear"]
    )

    df["Battery_Temp_Rolling"] = (
        df["Battery_Temperature"]
        .rolling(window=5)
        .mean()
    )

    df["Motor_Vibration_STD"] = (
        df["Motor_Vibration"]
        .rolling(window=5)
        .std()
    )

    df = df.bfill()

    return df
