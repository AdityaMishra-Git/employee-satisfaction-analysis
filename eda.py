# eda.py
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def perform_eda(df):
    # Summary statistics
    print("\nSummary Statistics:\n", df.describe())

    # Visualizations
    plt.figure(figsize=(10, 6))
    sb.histplot(df['satisfaction_level'], bins=20, kde=True)
    plt.title('Distribution of Satisfaction Level')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Frequency')
    plt.savefig('satisfaction_distribution.png')  
    plt.show()

    # Boxplot of satisfaction level by department
    plt.figure(figsize=(12, 6))
    sb.boxplot(x='dept', y='satisfaction_level', data=df)
    plt.title('Satisfaction Level by Department')
    plt.xticks(rotation=45)
    plt.savefig('satisfaction_by_department.png')  
    plt.show()

if __name__ == "__main__":
    # Assuming cleaned_data is imported or passed here
    pass
