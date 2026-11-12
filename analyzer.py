import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_house_data(df):
    print("\n========== DATA PREPROCESSING ==========")

    X = df[
        ["area_sqft", "bedrooms", "bathrooms", "age_years", "location"]
    ]

    y = df["price_lakhs"]

    numeric_features = [
        "area_sqft", "bedrooms", "bathrooms", "age_years"
    ]

    categorical_features = ["location"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("location_encoder",
             OneHotEncoder(handle_unknown="ignore"),
             categorical_features)
        ],
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print("Preprocessor created successfully.")

    return X_train, X_test, y_train, y_test, preprocessor


def load_house_data(file_path):

    df = pd.read_csv(file_path)

    required_columns = [
        "house_id",
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "age_years",
        "location",
        "price_lakhs"
    ]

    # Check required columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Required columns are missing.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Check missing values
    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values found.")

    # Convert numerical columns
    numeric_columns = [
        "house_id",
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "age_years",
        "price_lakhs"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Validate numerical values
    if not np.isfinite(df[numeric_columns].to_numpy()).all():
        raise ValueError("Invalid numerical values found.")

    if (df[numeric_columns] < 0).any().any():
        raise ValueError("Negative values are not allowed.")

    if (df[["area_sqft", "bedrooms", "bathrooms", "price_lakhs"]] == 0).any().any():
        raise ValueError("Area, rooms and price must be greater than zero.")

    if df["house_id"].duplicated().any():
        raise ValueError("Duplicate house IDs found.")

    valid_locations = {"Suburban", "Urban", "Premium"}

    if not set(df["location"]).issubset(valid_locations):
        raise ValueError("Invalid property location found.")

    print("House dataset loaded successfully!")
    print(f"Total properties: {len(df)}")

    return df



def analyze_house_data(df):
    print("\n========== HOUSE PRICE ANALYSIS ==========")

    print("\nDataset Overview:")
    print(f"Total Houses: {len(df)}")
    print(f"Average Price: ₹{df['price_lakhs'].mean():.2f} Lakhs")
    print(f"Median Price: ₹{df['price_lakhs'].median():.2f} Lakhs")
    print(f"Minimum Price: ₹{df['price_lakhs'].min():.2f} Lakhs")
    print(f"Maximum Price: ₹{df['price_lakhs'].max():.2f} Lakhs")

    print("\nAverage Price by Location:")
    location_prices = df.groupby("location")["price_lakhs"].mean()
    print(location_prices.round(2))

    print("\nAverage Area by Location:")
    location_area = df.groupby("location")["area_sqft"].mean()
    print(location_area.round(2))

    print("\nFeature Correlation with House Price:")
    correlations = df.select_dtypes(include="number").corr()
    print(correlations["price_lakhs"].sort_values(ascending=False))



def train_house_price_model(X_train, y_train, preprocessor):
    print("\n========== MODEL TRAINING ==========")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ))
        ]
    )

    model.fit(X_train, y_train)

    print("Random Forest model trained successfully!")

    return model


def evaluate_house_model(model, X_test, y_test):
    print("\n========== MODEL EVALUATION ==========")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae:.2f} Lakhs")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f} Lakhs")
    print(f"R2 Score: {r2:.4f}")

    return predictions


def predict_house_price(model, area, bedrooms, bathrooms, age, location):
    print("\n========== NEW HOUSE PREDICTION ==========")

    new_house = pd.DataFrame([{
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "location": location
    }])

    predicted_price = model.predict(new_house)[0]

    print(f"Area: {area} sq.ft.")
    print(f"Bedrooms: {bedrooms}")
    print(f"Bathrooms: {bathrooms}")
    print(f"Age: {age} years")
    print(f"Location: {location}")
    print(f"Predicted Price: ₹{predicted_price:.2f} Lakhs")

    return predicted_price



def visualize_predictions(y_test, predictions):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(len(y_test)),
        y_test.values,
        marker="o",
        label="Actual Prices"
    )

    plt.plot(
        range(len(predictions)),
        predictions,
        marker="s",
        label="Predicted Prices"
    )

    plt.xlabel("Test House")
    plt.ylabel("Price (Lakhs)")
    plt.title("Actual vs Predicted House Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("price_comparison.png")
    plt.show()

    print("Chart saved as price_comparison.png")


def export_predictions(y_test, predictions):
    results = pd.DataFrame({
        "actual_price": y_test.values,
        "predicted_price": predictions
    })

    results["absolute_error"] = abs(
        results["actual_price"] - results["predicted_price"]
    )

    results.to_csv("house_predictions.csv", index=False)

    print("\nPrediction report exported successfully!")
    print("File: house_predictions.csv")

