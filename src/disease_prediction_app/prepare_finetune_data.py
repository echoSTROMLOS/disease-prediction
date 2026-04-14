import pandas as pd
import json

df = pd.read_csv("synthetic_data/synthetic_pathological_data.csv")

def create_instruction(row):
    instruction = (
        f"Patient Information:\n"
        f"Age: {row['age']}\n"
        f"Gender: {row['gender']}\n"
        f"Symptoms: {row['symptoms']}\n"
        f"Temperature: {row['temperature_c']} °C\n"
        f"Blood Pressure: {row['blood_pressure']}\n"
        f"Heart Rate: {row['heart_rate_bpm']} bpm\n"
        f"Respiratory Rate: {row['respiratory_rate']}\n"
        f"SpO2: {row['spo2']}%\n"
        f"WBC Count: {row['wbc_count']} x10^9/L\n"
        f"Glucose: {row['glucose']} mg/dL\n"
        f"HbA1c: {row['hba1c']}%\n"
        f"Medical History: {row['medical_history']}\n\n"
        f"Question: What is the most likely diagnosis and recommended treatment?"
    )

    output = (
        f"Diagnosis: {row['diagnosis']}\n"
        f"Severity: {row['severity']}\n"
        f"Recommended Tests: {row['recommended_tests']}\n"
        f"Treatment Plan: {row['treatment_plan']}"
    )

    return {
        "instruction": instruction,
        "input": "",
        "output": output
    }

dataset = [create_instruction(row) for _, row in df.iterrows()]

with open("synthetic_data/finetune_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)