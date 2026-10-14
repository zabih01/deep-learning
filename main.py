
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------
# 1. Load the dataset
# ----------------------------------
file_path = Path(__file__).parent / "data" / "salary_dataset.xlsx"
df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print(df.head())

# ----------------------------------
# 2. Prepare input and target
# ----------------------------------
X = df[["YearsExperience"]]
y = df["Salary"]

# ----------------------------------
# 3. Train the model
# ----------------------------------
model = LinearRegression()
model.fit(X, y)

print("\nModel training completed!")

# ----------------------------------
# 4. Answer Q1: Predict salary
# ----------------------------------
new_employee = pd.DataFrame({
    "YearsExperience": [12]
})

predicted_salary = model.predict(new_employee)[0]

print("\n--- Q1: Salary Prediction ---")
print("Years of Experience: 12")
print(f"Predicted Salary: {predicted_salary:.2f}")

# ----------------------------------
# 5. Answer Q2: Regression equation
# ----------------------------------
intercept = model.intercept_
coefficient = model.coef_[0]

print("\n--- Q2: Regression Equation ---")
print(f"Intercept (b0): {intercept:.6f}")
print(f"Coefficient (b1): {coefficient:.6f}")

print(
    f"Salary = {intercept:.2f} "
    f"+ ({coefficient:.2f} * YearsExperience)"
)

# ----------------------------------
# 6. Evaluate the model
# ----------------------------------
predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
rmse = mse ** 0.5
r2 = r2_score(y, predictions)

print("\n--- Model Evaluation ---")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Training R-squared: {r2:.4f}")

# ----------------------------------
# 7. Visualize actual data and line
# ----------------------------------
plt.scatter(
    X["YearsExperience"],
    y,
    label="Actual Salary"
)

plt.plot(
    X["YearsExperience"],
    predictions,
    color="red",
    label="Regression Line"
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction Using Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the graph in the project folder
plot_path = Path(__file__).parent / "regression_plot.png"
plt.savefig(plot_path, dpi=150)

print(f"\nGraph saved to: {plot_path}")

plt.show()
