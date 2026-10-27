"""
Campus Placement Predictor CLI
Author: Aryaman Srimali Jain (Zenecks Infotech ML Internship)
"""

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "placement_predictor_pipeline.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def predict_placement(gender="M", ssc_p=72.0, ssc_b="Central", hsc_p=75.0, hsc_b="Central",
                      hsc_s="Science", degree_p=70.0, degree_t="Sci&Tech", workex="Yes",
                      etest_p=78.0, specialisation="Mkt&Fin", mba_p=68.0):
    artifact = load_model()
    model = artifact["model"]
    scaler = artifact["scaler"]
    encoders = artifact["label_encoders"]

    # Transform categorical values
    gender_enc = encoders["gender"].transform([gender])[0]
    ssc_b_enc = encoders["ssc_b"].transform([ssc_b])[0]
    hsc_b_enc = encoders["hsc_b"].transform([hsc_b])[0]
    hsc_s_enc = encoders["hsc_s"].transform([hsc_s])[0]
    degree_t_enc = encoders["degree_t"].transform([degree_t])[0]
    workex_enc = encoders["workex"].transform([workex])[0]
    spec_enc = encoders["specialisation"].transform([specialisation])[0]

    feature_vector = np.array([[
        gender_enc, ssc_p, ssc_b_enc, hsc_p, hsc_b_enc, hsc_s_enc,
        degree_p, degree_t_enc, workex_enc, etest_p, spec_enc, mba_p
    ]])

    scaled_vector = scaler.transform(feature_vector)
    prob_placed = model.predict_proba(scaled_vector)[0][1]
    is_placed = prob_placed >= 0.50

    return {
        "status": "Placed" if is_placed else "Not Placed",
        "probability_placed": float(prob_placed),
        "confidence": float(prob_placed if is_placed else (1.0 - prob_placed)),
        "input_profile": {
            "10th Grade (SSC)": f"{ssc_p}%",
            "12th Grade (HSC)": f"{hsc_p}%",
            "Undergrad Degree": f"{degree_p}% ({degree_t})",
            "Work Experience": workex,
            "Employability Test": f"{etest_p}%",
            "MBA Specialisation": specialisation,
            "MBA Score": f"{mba_p}%"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Campus Placement Outcome for a Candidate Profile")
    parser.add_argument("--gender", default="M", choices=["M", "F"], help="Candidate Gender")
    parser.add_argument("--ssc_p", type=float, default=74.0, help="10th Grade Percentage (SSC)")
    parser.add_argument("--ssc_b", default="Central", choices=["Central", "Others"], help="10th Board")
    parser.add_argument("--hsc_p", type=float, default=78.0, help="12th Grade Percentage (HSC)")
    parser.add_argument("--hsc_b", default="Central", choices=["Central", "Others"], help="12th Board")
    parser.add_argument("--hsc_s", default="Science", choices=["Commerce", "Science", "Arts"], help="12th Stream")
    parser.add_argument("--degree_p", type=float, default=72.0, help="Undergraduate Degree Percentage")
    parser.add_argument("--degree_t", default="Sci&Tech", choices=["Sci&Tech", "Comm&Mgmt", "Others"], help="Degree Stream")
    parser.add_argument("--workex", default="Yes", choices=["Yes", "No"], help="Prior Work Experience")
    parser.add_argument("--etest_p", type=float, default=80.0, help="Employability Test Percentage")
    parser.add_argument("--specialisation", default="Mkt&Fin", choices=["Mkt&Fin", "Mkt&HR"], help="MBA Specialization")
    parser.add_argument("--mba_p", type=float, default=68.5, help="MBA Percentage")

    args = parser.parse_args()
    res = predict_placement(
        gender=args.gender, ssc_p=args.ssc_p, ssc_b=args.ssc_b,
        hsc_p=args.hsc_p, hsc_b=args.hsc_b, hsc_s=args.hsc_s,
        degree_p=args.degree_p, degree_t=args.degree_t, workex=args.workex,
        etest_p=args.etest_p, specialisation=args.specialisation, mba_p=args.mba_p
    )

    print("=" * 60)
    print("ZENECKS INFOTECH - CAMPUS RECRUITMENT PREDICTION ENGINE")
    print("=" * 60)
    print("Candidate Profile:")
    for k, v in res["input_profile"].items():
        print(f"  * {k:22s}: {v}")
    print("-" * 60)
    verdict_tag = "[PASS]" if res["status"] == "Placed" else "[FAIL]"
    print(f"Prediction Verdict    : {verdict_tag} {res['status'].upper()}")
    print(f"Placement Probability : {res['probability_placed']*100:.2f}%")
    print(f"Model Confidence      : {res['confidence']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
