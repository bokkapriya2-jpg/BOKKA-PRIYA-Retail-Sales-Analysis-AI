# BOKKA_PRIYA

**Author:** BOKKA_PRIYA  
**Program:** IBM SkillsBuild Data Analytics with AI Academic Internship  
**Project:** Retail Sales Analysis and AI-Based Sales Prediction

## Project Description
This project performs retail sales data analysis and builds a machine-learning model to predict sales for a transaction.

The notebook creates a reproducible synthetic retail dataset with 1,500 transactions. It then performs data cleaning checks, exploratory data analysis, visualization, business insights, and AI-based sales prediction using a Random Forest Regressor.

## Dataset
The dataset is generated programmatically inside the notebook, so no external dataset download is required.

**Dataset size:** 1,500 transactions  
**Time period:** January 2025 to December 2025

Main columns:
- Date
- Category
- Region
- Customer_Segment
- Quantity
- Discount
- Unit_Price
- Sales
- Profit
- Month

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## AI/ML Method
A **Random Forest Regression** model is used to predict transaction-level Sales from categorical and numerical features.

Evaluation metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

## How to Run

1. Install Python 3.10 or later.
2. Open a terminal in the project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `BOKKA_PRIYA_Retail_Sales_Analysis_AI.ipynb`.
6. Run all cells from top to bottom.

## Project Outputs
The notebook produces:
- Monthly sales trend
- Category-wise sales analysis
- Region-wise profit analysis
- Customer-segment analysis
- Correlation heatmap
- Business KPIs
- Actual vs predicted sales chart
- Feature importance chart
- Example AI sales prediction

## Key Outcome
The project demonstrates a complete data analytics workflow: data generation, data validation, exploratory analysis, visualization, business insight extraction, machine-learning model development, evaluation, and prediction.
