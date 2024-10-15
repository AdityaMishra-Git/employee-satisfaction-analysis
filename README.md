# Employee Satisfaction Analysis

## Project Overview
This project aims to analyze employee satisfaction using a dataset containing various employee metrics. The analysis includes exploratory data analysis (EDA), data cleaning, hypothesis testing, and clustering to identify satisfaction patterns.

## Project Structure
The project is organized into the following modules:
- **data_cleaning.py**: Contains functions for cleaning and preprocessing the dataset.
- **eda.py**: Performs exploratory data analysis and visualizations.
- **hypothesis_testing.py**: Conducts statistical tests to validate assumptions.
- **clustering.py**: Implements clustering algorithms, including K-Means, to group employees based on satisfaction levels.
- **visualization.py**: Contains functions to generate visual representations of the data.
- **main.py**: The entry point for running the analysis. This file orchestrates the execution of other modules.
- **employee_analysis.py**: Contains the full code if you wish to view or run the entire process at once.

## DataSet 
The dataset used for this analysis is employee_data.csv, which contains several key features to analyze employee satisfaction. Below are the details of each column:

- **EmployeeID**: Unique identifier for each employee.
- **Department**: The department to which the employee belongs (e.g., HR, Sales, IT).
- **Satisfaction**: Satisfaction rating of the employee on a scale of 1 to 10.
- **Hours**: Number of hours worked by the employee in a week.
- **Projects**: Number of projects the employee is involved in.
- **Tenure**: Number of years the employee has been with the company.
- **Salary**: The salary level of the employee (Low, Medium, High).
- **Performance Rating**: A score representing the employee's performance rating.
- **Age**: Age of the employee.

  ## Dataset Usage
   The dataset is used for the following analyses:

- Identifying satisfaction distribution across departments.
- Exploring correlations between satisfaction and various factors such as working hours and tenure.
- Conducting hypothesis testing to validate assumptions about employee satisfaction.
- Grouping employees based on satisfaction levels using clustering algorithms.

## Requirements
To run this project, ensure you have the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

You can install the requirements using:

```bash
pip install -r requirements.txt


