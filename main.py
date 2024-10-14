# main.py
import pandas as pd
from data_cleaning import load_data, clean_data
from eda import perform_eda
from hypothesis_testing import perform_hypothesis_testing
from clustering import perform_clustering
from visualization import visualize_correlation

def main():
    # Load and clean data
    df = load_data('employee_data.csv')
    cleaned_data = clean_data(df)
    
    # Perform EDA
    perform_eda(cleaned_data)

    # Perform hypothesis testing
    perform_hypothesis_testing(cleaned_data)

    # Perform clustering
    perform_clustering(cleaned_data)

    # Visualize correlation
    visualize_correlation(cleaned_data)

if __name__ == "__main__":
    main()
