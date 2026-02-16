# Linear-regression-
This project demonstrates a simple linear regression model built using Python to predict house prices based on house size. The main objective is to understand how linear regression works, how to train a model using real data, and how to evaluate its performance using standard metrics. The dataset contains two columns: house size as the independent # Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 2: Load the dataset
# Example: House Price dataset with Size vs Price
data = {
    'Size': [800, 1000, 1200, 1500, 1800, 2000],
    'Price': [120000, 150000, 180000, 220000, 260000, 300000]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# Step 3: Define features (X) and target (y)
X = df[['Size']]   # Independent variable
y = df['Price']    # Dependent variable


# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 5: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Step 6: Make predictions
y_pred = model.predict(X_test)


# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)


# Step 8: Print model coefficients
print("\nModel Parameters:")
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)


# Step 9: Plot regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
