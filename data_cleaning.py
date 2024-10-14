# data_cleaning.py
import pandas as pd
import numpy as np
from scipy import stats

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values
    df['satisfaction_level'] = df['satisfaction_level'].fillna(df['satisfaction_level'].mean())
    df['last_evaluation'] = df['last_evaluation'].fillna(df['last_evaluation'].mean())
    df['number_project'] = df['number_project'].fillna(df['number_project'].median())
    df['average_montly_hours'] = df['average_montly_hours'].fillna(df['average_montly_hours'].mean())
    df['time_spend_company'] = df['time_spend_company'].fillna(df['time_spend_company'].median())

    # Check if 'work_accident' exists before filling
    if 'work_accident' in df.columns:
        df['work_accident'] = df['work_accident'].fillna(df['work_accident'].mode()[0])

    # Similarly for other columns...
    if 'promotion_last_5years' in df.columns:
        df['promotion_last_5years'] = df['promotion_last_5years'].fillna(df['promotion_last_5years'].mode()[0])
    
    # Remove Outliers using Z-score
    z_scores = np.abs(stats.zscore(df[['satisfaction_level', 'last_evaluation', 
                                          'number_project', 'average_montly_hours', 
                                          'time_spend_company']]))
    df_no_outliers = df[(z_scores < 3).all(axis=1)].copy() 

    return df_no_outliers

if __name__ == "__main__":
    df = load_data('employee_data.csv')
    cleaned_data = clean_data(df)
    print(cleaned_data.head())
