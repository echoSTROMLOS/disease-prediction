import pandas as pd
import random
import uuid
from faker import Faker

fake = Faker()

# Disease profiles
diseases = {
    "Influenza": {
        "symptoms": ["fever", "cough", "sore throat", "body aches", "fatigue"],
        "tests": ["Rapid Influenza Diagnostic Test", "PCR Test"],
        "treatment": "Rest, hydration, and antiviral medications like oseltamivir."
    },
    "COVID-19": {
        "symptoms": ["fever", "dry cough", "shortness of breath", "loss of taste", "fatigue"],
        "tests": ["RT-PCR", "Antigen Test", "Chest CT"],
        "treatment": "Supportive care, oxygen therapy, and antivirals."
    },
    "Pneumonia": {
        "symptoms": ["fever", "productive cough", "chest pain", "shortness of breath"],
        "tests": ["Chest X-ray", "Sputum Culture", "CBC"],
        "treatment": "Antibiotics and supportive care."
    },
    "Diabetes Mellitus Type 2": {
        "symptoms": ["increased thirst", "frequent urination", "fatigue", "blurred vision"],
        "tests": ["Fasting Blood Glucose", "HbA1c"],
        "treatment": "Lifestyle changes and medications like metformin."
    },
    "Hypertension": {
        "symptoms": ["headache", "dizziness", "blurred vision"],
        "tests": ["Blood Pressure Monitoring", "Lipid Profile"],
        "treatment": "Lifestyle modification and antihypertensive medications."
    },
    "Asthma": {
        "symptoms": ["wheezing", "shortness of breath", "chest tightness", "cough"],
        "tests": ["Spirometry", "Peak Flow Measurement"],
        "treatment": "Inhaled bronchodilators and corticosteroids."
    },
    "Tuberculosis": {
        "symptoms": ["chronic cough", "night sweats", "weight loss", "fever"],
        "tests": ["Sputum AFB", "Chest X-ray", "Mantoux Test"],
        "treatment": "Combination anti-tubercular therapy."
    },
    "Dengue Fever": {
        "symptoms": ["high fever", "severe headache", "joint pain", "rash"],
        "tests": ["NS1 Antigen", "Dengue IgM/IgG"],
        "treatment": "Fluid management and supportive care."
    },
    "Malaria": {
        "symptoms": ["fever with chills", "sweating", "headache", "nausea"],
        "tests": ["Peripheral Blood Smear", "Rapid Diagnostic Test"],
        "treatment": "Antimalarial medications such as artemisinin-based therapies."
    },
    "Urinary Tract Infection": {
        "symptoms": ["burning urination", "frequent urination", "pelvic pain"],
        "tests": ["Urinalysis", "Urine Culture"],
        "treatment": "Antibiotics and hydration."
    }
}

def generate_vitals(disease):
    temp = round(random.uniform(36.5, 39.5), 1) if disease not in ["Hypertension", "Diabetes Mellitus Type 2"] else round(random.uniform(36.0, 37.2), 1)
    return {
        "temperature_c": temp,
        "blood_pressure": f"{random.randint(110, 160)}/{random.randint(70, 100)}",
        "heart_rate_bpm": random.randint(60, 120),
        "respiratory_rate": random.randint(12, 30),
        "spo2": random.randint(90, 99)
    }

def generate_lab_results(disease):
    return {
        "wbc_count": round(random.uniform(4.0, 15.0), 2),
        "hemoglobin": round(random.uniform(10.0, 17.0), 1),
        "platelets": random.randint(50000, 450000),
        "glucose": random.randint(70, 250),
        "hba1c": round(random.uniform(4.5, 10.0), 1)
    }

def generate_record():
    disease = random.choice(list(diseases.keys()))
    profile = diseases[disease]
    symptoms = random.sample(profile["symptoms"], k=random.randint(2, len(profile["symptoms"])))

    record = {
        "patient_id": str(uuid.uuid4()),
        "age": random.randint(1, 90),
        "gender": random.choice(["Male", "Female"]),
        "symptoms": ", ".join(symptoms),
        "medical_history": fake.sentence(nb_words=6),
        "diagnosis": disease,
        "severity": random.choice(["Mild", "Moderate", "Severe"]),
        "recommended_tests": ", ".join(profile["tests"]),
        "treatment_plan": profile["treatment"],
    }

    record.update(generate_vitals(disease))
    record.update(generate_lab_results(disease))
    return record

def generate_dataset(n_samples=5000, output_file="synthetic_data/synthetic_pathological_data.csv"):
    data = [generate_record() for _ in range(n_samples)]
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()