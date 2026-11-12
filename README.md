# 🏠 House Price Prediction System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Model-red?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

The **House Price Prediction System** is a Machine Learning project that predicts house prices based on property characteristics such as area, number of bedrooms, bathrooms, property age, and location.

The project uses a **Random Forest Regression** model to learn patterns from housing data and estimate the price of new properties.

It also performs exploratory data analysis, evaluates model performance, visualizes predictions, and exports prediction results into a CSV report.

## 🎯 Project Objectives

- Analyze housing datasets using Python.
- Understand relationships between property features and prices.
- Convert categorical location data into numerical features.
- Train a Random Forest Regression model.
- Evaluate predictions using regression metrics.
- Predict prices for new houses.
- Visualize and export prediction results.

## ⚙️ Features

- CSV dataset loading and validation.
- House price statistical analysis.
- Location-wise price analysis.
- Numerical feature correlation analysis.
- One-Hot Encoding for categorical data.
- Train-test splitting.
- Random Forest Regression model.
- Model evaluation using MAE, MSE, RMSE, and R².
- New house price prediction.
- Actual vs predicted price visualization.
- CSV prediction report generation.

## 🧠 Machine Learning Workflow

1. Load the housing dataset.
2. Validate and clean the data.
3. Perform exploratory data analysis.
4. Separate input features (X) and target (y).
5. Encode categorical location data using One-Hot Encoding.
6. Split the dataset into training and testing sets.
7. Train a Random Forest Regression model.
8. Evaluate the model using regression metrics.
9. Predict prices for new properties.
10. Generate visualizations and export prediction results.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning |
| Random Forest | Price prediction algorithm |

## 📂 Project Structure

```text
house-price-predictor/
│
├── main.py
├── analyzer.py
├── house_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset Information

The project uses a sample housing dataset containing 30 records.

| Feature | Description |
|---|---|
| house_id | Unique property identifier |
| area_sqft | Property area in square feet |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| age_years | Age of the property |
| location | Suburban, Urban, or Premium |
| price_lakhs | House price in lakhs |

**Target Variable:** `price_lakhs`

**Input Features:** Area, bedrooms, bathrooms, age, and location.

## 📈 Model Evaluation

The model is evaluated using:

- **MAE (Mean Absolute Error):** Measures average absolute prediction error.
- **MSE (Mean Squared Error):** Penalizes larger prediction errors.
- **RMSE (Root Mean Squared Error):** Measures prediction error in the original price unit.
- **R² Score:** Measures how much variation in house prices is explained by the model.

Lower MAE, MSE, and RMSE indicate smaller errors. An R² score closer to 1 generally indicates a better fit.

## 🖥️ Example Prediction

Example input:

```python
area = 1400
bedrooms = 3
bathrooms = 2
age = 5
location = "Urban"
```

The trained model processes these features and returns an estimated house price in lakhs.

The exact prediction depends on the trained model and dataset.

## 📁 Generated Files

| File | Description |
|---|---|
| price_comparison.png | Actual vs predicted price chart |
| house_predictions.csv | Prediction results and absolute errors |

## 🚀 Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/ayushman-ece/house-price-predictor.git
```

**2. Open the project folder**

```bash
cd house-price-predictor
```

**3. Create a virtual environment (optional)**

```bash
python -m venv .venv
```

**4. Activate the environment**

Windows:

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the project**

```bash
python main.py
```

## 📦 Requirements

```text
pandas
numpy
scikit-learn
matplotlib
```

## 📚 Key Concepts Learned

- Exploratory Data Analysis (EDA)
- Data validation and preprocessing
- Feature selection
- One-Hot Encoding
- Train-test splitting
- Random Forest Regression
- Scikit-learn Pipelines
- Regression evaluation metrics
- Model prediction
- Data visualization
- CSV report generation

## ⚠️ Limitations

- The dataset contains only 30 sample records.
- The dataset is synthetic and does not represent actual real-estate market data.
- Model performance may not generalize to real-world properties.
- Important factors such as property condition, amenities, neighbourhood quality, and market trends are not included.

**This project is intended for learning Machine Learning concepts and should not be used for actual property valuation.**

## 🔮 Future Improvements

- Use a larger real-world housing dataset.
- Include additional property features.
- Compare Random Forest with Linear Regression and Gradient Boosting.
- Perform hyperparameter tuning.
- Build a user-friendly web application using Streamlit.
- Deploy the model as an API.

## 👨‍💻 Author

**Ayushman Tiwari**

GitHub: [@ayushman-ece](https://github.com/ayushman-ece)

---

⭐ If you find this project useful, consider giving the repository a star!