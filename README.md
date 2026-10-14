# Salary Prediction Using Linear Regression

## Project Overview

This project demonstrates how to build and use a **Linear Regression** machine learning model to predict an employee's salary based on their years of experience.

The project uses Python and popular data science libraries to train a regression model, make predictions, and extract the final linear regression equation.

## Objectives

This project solves the following questions:

### Q1: Making a Single Prediction

Predict the salary of an employee who has **12 years of experience** using the trained Linear Regression model.

The model takes the employee's years of experience as input and predicts the corresponding salary.

### Q2: Getting the Final Linear Regression Equation

Extract the coefficient (slope) and intercept from the trained model to construct the final Linear Regression equation.

The equation is:

`Salary = b₀ + b₁ × Years of Experience`

Where:

- **Salary:** Predicted employee salary.
- **b₀ (Intercept):** The predicted salary when years of experience is zero.
- **b₁ (Coefficient):** The estimated change in salary for each additional year of experience.

The actual coefficient and intercept values are calculated after training the model.

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- openpyxl
- uv (Python package and project manager)

## Project Structure

```text
deep-learning/
├── data/
│   └── salary_dataset.xlsx
├── main.py
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

The `.python-version` file is optional. Your project may not contain it.

## Dataset

The project uses an Excel dataset named `salary_dataset.xlsx`, stored inside the `data` folder.

The dataset should contain the relevant columns for:

- Years of Experience
- Salary

Ensure the dataset column names match those used in `main.py`.

## Installation and Setup

### 1. Install Python and uv

Install Python and the `uv` package manager if they are not already installed.

### 2. Open the Project Directory

```bash
cd C:\Users\Admin\Desktop\deep-learning
```

### 3. Install Project Dependencies

If the dependencies are already defined in `pyproject.toml`, run:

```bash
uv sync
```

### 4. Run the Project

```bash
uv run main.py
```

## Solutions

### Q1: Predict Salary for 12 Years of Experience

Use the trained model to make a prediction:

```python
prediction = model.predict([[12]])
print("Predicted Salary:", prediction[0])
```

This predicts the salary for an employee with 12 years of experience.

**Note:** The example assumes that the model was trained using years of experience as its single input feature.

### Q2: Obtain the Final Linear Regression Equation

Retrieve the coefficient and intercept:

```python
coefficient = model.coef_[0]
intercept = model.intercept_

print("Coefficient:", coefficient)
print("Intercept:", intercept)

print(
    f"Salary = {intercept:.2f} + "
    f"({coefficient:.2f} × Years of Experience)"
)
```

The final equation has the form:

`Salary = Intercept + (Coefficient × Years of Experience)`

The numerical equation depends on the trained model and dataset.

## Expected Results

After running the project, the program should produce:

1. The predicted salary for an employee with 12 years of experience.
2. The trained model's coefficient.
3. The trained model's intercept.
4. The final Linear Regression equation.

The exact numerical results depend on the dataset and training process.

## Conclusion

This project demonstrates two fundamental machine learning tasks using Linear Regression: making a prediction for a new data point and interpreting the trained model through its coefficient, intercept, and equation.

It provides a practical introduction to regression modeling with Python and scikit-learn.
