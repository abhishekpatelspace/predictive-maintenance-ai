from kafka import KafkaProducer
import pandas as pd
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

df = pd.read_csv("../data/EV_Predictv_data.csv")

for _, row in df.iterrows():

    producer.send(
        'telemetry_topic',
        row.to_dict()
    )

    print("Sent:", row.to_dict())

    time.sleep(1)
