# visualization.py
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def visualize_correlation(df):
    correlation_matrix = df[['satisfaction_level', 'last_evaluation', 'average_montly_hours', 'time_spend_company']].corr()
    
    plt.figure(figsize=(8, 6))
    sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')  
    plt.show()

if __name__ == "__main__":
    # Assuming cleaned_data is imported or passed here
    pass
