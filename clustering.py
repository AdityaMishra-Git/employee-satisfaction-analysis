# clustering.py
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def perform_clustering(df):
    features = df[['satisfaction_level', 'last_evaluation', 'average_montly_hours']]
    kmeans = KMeans(n_clusters=3)  
    df['cluster'] = kmeans.fit_predict(features)

    # Plot clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(df['average_montly_hours'], df['satisfaction_level'], c=df['cluster'], cmap='viridis')
    plt.title('K-Means Clustering of Employees')
    plt.xlabel('Average Monthly Hours')
    plt.ylabel('Satisfaction Level')
    plt.savefig('kmeans_clustering.png')  
    plt.show()

if __name__ == "__main__":
    # Assuming cleaned_data is imported or passed here
    pass
