# =============================================================
# predict.py
# Purpose : Standalone prediction script.
#           Loads saved model.pkl + encoder.pkl and predicts
#           the selling price for a new car input.
# Usage   : python predict.py
# =============================================================

import pickle
import numpy as np
import pandas as pd
import os
import sys


# ---------------------------------------------
# Load Artifacts
# ---------------------------------------------
def load_artifacts(model_path: str = "model.pkl",
                   encoder_path: str = "encoder.pkl"):
    """
    Load the trained model and OneHotEncoder from disk.

    Parameters
    ----------
    model_path   : Path to the pickled model file.
    encoder_path : Path to the pickled encoder file.

    Returns
    -------
    (model, encoder)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Please run 'python train_model.py' first."
        )
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Encoder not found at '{encoder_path}'.\n"
            "Please run 'python train_model.py' first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    print(f"[INFO] Model   loaded from : {model_path}")
    print(f"[INFO] Encoder loaded from : {encoder_path}\n")
    return model, encoder


# ---------------------------------------------
# Preprocess Single Input
# ---------------------------------------------
def preprocess_input(
    present_price: float,
    car_age: int,
    kms_driven: float,
    fuel_type: str,
    seller_type: str,
    transmission: str,
    owner: int,
    encoder,
) -> np.ndarray:
    """
    Transform raw user inputs into the feature vector expected by the model.

    Parameters
    ----------
    present_price : Showroom price in Rs. Lakhs.
    car_age       : Age of car in years.
    kms_driven    : Total kilometres driven.
    fuel_type     : 'Petrol', 'Diesel', or 'CNG'.
    seller_type   : 'Dealer' or 'Individual'.
    transmission  : 'Manual' or 'Automatic'.
    owner         : Number of previous owners (0, 1, 2, 3).
    encoder       : Fitted OneHotEncoder.

    Returns
    -------
    np.ndarray : Feature array ready for model.predict().
    """
    # Build categorical part
    cat_input = pd.DataFrame([[fuel_type, seller_type, transmission]],
                             columns=["Fuel_Type", "Seller_Type", "Transmission"])
    encoded_cats = encoder.transform(cat_input)

    # Build numeric part — ORDER MUST MATCH TRAINING COLUMN ORDER:
    # Present_Price, Kms_Driven, Owner, Car_Age  (from Step 5 output)
    numeric = np.array([[present_price, kms_driven, owner, car_age]])

    # Concatenate: numeric first, then encoded categoricals
    # Must match the column order used during training.
    # Training order: Present_Price, Kms_Driven, Owner, Car_Age, <encoded_cats>
    features = np.concatenate([numeric, encoded_cats], axis=1)
    return features


# ---------------------------------------------
# Predict Price
# ---------------------------------------------
def predict_price(
    present_price: float,
    car_age: int,
    kms_driven: float,
    fuel_type: str,
    seller_type: str,
    transmission: str,
    owner: int,
    model=None,
    encoder=None,
    model_path: str = "model.pkl",
    encoder_path: str = "encoder.pkl",
) -> float:
    """
    End-to-end prediction function.

    If model and encoder are not provided, loads them from disk.

    Returns
    -------
    float : Predicted selling price in Rs. Lakhs.
    """
    if model is None or encoder is None:
        model, encoder = load_artifacts(model_path, encoder_path)

    features = preprocess_input(
        present_price, car_age, kms_driven,
        fuel_type, seller_type, transmission, owner,
        encoder,
    )

    price = model.predict(features)[0]
    # Clip to non-negative (can't have negative price)
    price = max(0.0, round(float(price), 2))
    return price


# ---------------------------------------------
# Interactive CLI Demo
# ---------------------------------------------
def run_demo() -> None:
    """
    Interactive command-line demo that takes user inputs and
    prints the predicted selling price.
    """
    print("\n" + "=" * 55)
    print("  CAR SELLING PRICE PREDICTOR - CLI Demo")
    print("=" * 55)

    try:
        present_price = float(input("  Showroom Price (Rs. Lakhs)    : "))
        car_age       = int(input("  Car Age (years)             : "))
        kms_driven    = float(input("  Kilometres Driven           : "))
        fuel_type     = input("  Fuel Type [Petrol/Diesel/CNG]: ").strip().capitalize()
        seller_type   = input("  Seller Type [Dealer/Individual]: ").strip().capitalize()
        transmission  = input("  Transmission [Manual/Automatic]: ").strip().capitalize()
        owner         = int(input("  Number of Previous Owners   : "))
    except ValueError as e:
        print(f"\n[ERROR] Invalid input: {e}")
        sys.exit(1)

    price = predict_price(
        present_price=present_price,
        car_age=car_age,
        kms_driven=kms_driven,
        fuel_type=fuel_type,
        seller_type=seller_type,
        transmission=transmission,
        owner=owner,
    )

    print("\n" + "-" * 55)
    print(f"  Estimated Selling Price : Rs. {price:.2f} Lakhs")
    print("-" * 55 + "\n")


# ---------------------------------------------
# Quick Batch Demo (hardcoded samples)
# ---------------------------------------------
def run_batch_demo() -> None:
    """
    Predict on a few hardcoded examples to verify the model works.
    """
    model, encoder = load_artifacts()

    test_cases = [
        # (present_price, car_age, kms_driven, fuel_type, seller_type, transmission, owner)
        (5.59, 11, 27000, "Petrol", "Dealer", "Manual", 0),
        (9.54, 12, 43000, "Diesel", "Dealer", "Manual", 0),
        (9.85,  8,  6900, "Petrol", "Dealer", "Manual", 0),
        (4.15, 14,  5200, "Petrol", "Dealer", "Manual", 0),
        (6.87, 11, 42450, "Diesel", "Dealer", "Manual", 0),
    ]

    print("\n" + "=" * 70)
    print("  BATCH PREDICTION DEMO")
    print("=" * 70)
    print(f"  {'Present_P':>10}  {'Age':>4}  {'Kms':>8}  {'Fuel':>7}  {'Seller':>10}  {'Trans':>10}  {'Owner':>5}  {'Predicted Rs.':>12}")
    print("  " + "-" * 66)

    for (pp, age, kms, fuel, seller, trans, owner) in test_cases:
        price = predict_price(pp, age, kms, fuel, seller, trans, owner,
                              model=model, encoder=encoder)
        print(f"  {pp:>10.2f}  {age:>4}  {kms:>8.0f}  {fuel:>7}  {seller:>10}  {trans:>10}  {owner:>5}  {price:>10.2f} L")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Change to batch demo for quick verification
    import sys
    if "--demo" in sys.argv:
        run_batch_demo()
    else:
        run_demo()
