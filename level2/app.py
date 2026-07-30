# =============================================================
# app.py
# Purpose : Flask web application for Car Price Prediction.
#           Serves the prediction form and returns the
#           estimated selling price to the user.
# Usage   : python app.py
#           Then open http://localhost:5000 in a browser.
# =============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# -- Ensure predict module is importable ----------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict_price, load_artifacts

# ---------------------------------------------
# Flask App Initialisation
# ---------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "car_price_prediction_secret_2024"

# ---------------------------------------------
# Load Model & Encoder Once at Startup
# ---------------------------------------------
MODEL_PATH   = "model.pkl"
ENCODER_PATH = "encoder.pkl"

try:
    model, encoder = load_artifacts(MODEL_PATH, ENCODER_PATH)
    print("[Flask] Model and encoder loaded successfully.")
except FileNotFoundError as err:
    print(f"\n[Flask ERROR] {err}")
    print("[Flask] Run 'python train_model.py' first, then restart app.py.\n")
    model, encoder = None, None


# ---------------------------------------------
# Helper - Parse + Validate Form Data
# ---------------------------------------------
def parse_form(form) -> dict:
    """
    Extract and validate form fields from the POST request.

    Returns
    -------
    dict : Parsed values, or raises ValueError on bad input.
    """
    present_price = float(form.get("present_price", 0))
    car_age       = int(form.get("car_age", 0))
    kms_driven    = float(form.get("kms_driven", 0))
    fuel_type     = form.get("fuel_type", "Petrol").strip()
    seller_type   = form.get("seller_type", "Dealer").strip()
    transmission  = form.get("transmission", "Manual").strip()
    owner         = int(form.get("owner", 0))

    # Validation
    if present_price <= 0:
        raise ValueError("Showroom price must be greater than 0.")
    if car_age < 0 or car_age > 50:
        raise ValueError("Car age must be between 0 and 50 years.")
    if kms_driven < 0:
        raise ValueError("Kilometres driven cannot be negative.")
    if fuel_type not in ("Petrol", "Diesel", "CNG"):
        raise ValueError("Invalid fuel type.")
    if seller_type not in ("Dealer", "Individual"):
        raise ValueError("Invalid seller type.")
    if transmission not in ("Manual", "Automatic"):
        raise ValueError("Invalid transmission type.")
    if owner not in (0, 1, 2, 3):
        raise ValueError("Owner must be 0, 1, 2, or 3.")

    return dict(
        present_price=present_price,
        car_age=car_age,
        kms_driven=kms_driven,
        fuel_type=fuel_type,
        seller_type=seller_type,
        transmission=transmission,
        owner=owner,
    )


# ---------------------------------------------
# Routes
# ---------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Render the main prediction form page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction request.

    Accepts both HTML form POST and JSON API calls.
    Returns the predicted price embedded in the rendered template,
    or as a JSON response for API clients.
    """
    # Check model is loaded
    if model is None or encoder is None:
        error_msg = "Model not loaded. Run train_model.py first."
        if request.is_json:
            return jsonify({"error": error_msg}), 503
        return render_template("index.html", error=error_msg)

    try:
        # -- Parse inputs -------------------------------------
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        params = parse_form(data)

        # -- Predict ------------------------------------------
        price = predict_price(
            present_price=params["present_price"],
            car_age=params["car_age"],
            kms_driven=params["kms_driven"],
            fuel_type=params["fuel_type"],
            seller_type=params["seller_type"],
            transmission=params["transmission"],
            owner=params["owner"],
            model=model,
            encoder=encoder,
        )

        # -- Format result ------------------------------------
        result = {
            "price":  f"{price:.2f}",
            "inputs": params,
        }

        if request.is_json:
            return jsonify({"predicted_price": price, "unit": "Lakhs (Rs.)"})

        return render_template("index.html", prediction=result)

    except ValueError as ve:
        error_msg = str(ve)
        if request.is_json:
            return jsonify({"error": error_msg}), 400
        return render_template("index.html", error=error_msg)

    except Exception as exc:
        error_msg = f"Prediction failed: {exc}"
        if request.is_json:
            return jsonify({"error": error_msg}), 500
        return render_template("index.html", error=error_msg)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic access.

    Expects JSON body:
    {
        "present_price": 9.85,
        "car_age": 8,
        "kms_driven": 6900,
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": 0
    }
    """
    if model is None or encoder is None:
        return jsonify({"error": "Model not ready."}), 503

    try:
        data = request.get_json(force=True)
        params = parse_form(data)
        price = predict_price(model=model, encoder=encoder, **params)
        return jsonify({
            "status": "success",
            "predicted_price": price,
            "unit": "Lakhs (INR)",
            "inputs": params,
        })
    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None,
    })


# ---------------------------------------------
# Entry Point
# ---------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Car Price Prediction - Flask App")
    print("  http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
