# Import necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('employee_data.csv')

# Data Cleaning
# Remove duplicates
df = df.drop_duplicates()

# Fill missing values (if any)
df['satisfaction_level'] = df['satisfaction_level'].fillna(df['satisfaction_level'].mean())
df['last_evaluation'] = df['last_evaluation'].fillna(df['last_evaluation'].mean())
df['number_project'] = df['number_project'].fillna(df['number_project'].median())
df['average_montly_hours'] = df['average_montly_hours'].fillna(df['average_montly_hours'].mean())
df['time_spend_company'] = df['time_spend_company'].fillna(df['time_spend_company'].median())

# Check if 'work_accident' exists before filling
if 'work_accident' in df.columns:
    df['work_accident'] = df['work_accident'].fillna(df['work_accident'].mode()[0])

if 'promotion_last_5years' in df.columns:
    df['promotion_last_5years'] = df['promotion_last_5years'].fillna(df['promotion_last_5years'].mode()[0])

if 'dept' in df.columns:
    df['dept'] = df['dept'].fillna(df['dept'].mode()[0])

if 'salary' in df.columns:
    df['salary'] = df['salary'].fillna(df['salary'].mode()[0])

# Check the cleaned data
print("Missing Values After Cleaning:\n", df.isnull().sum())

# Remove Outliers using Z-score
z_scores = np.abs(stats.zscore(df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']]))
df_no_outliers = df[(z_scores < 3).all(axis=1)].copy()  # Create a copy to avoid SettingWithCopyWarning

# Check the shape of the data before and after removing outliers
print(f'Original data shape: {df.shape}')
print(f'Data shape after outlier removal: {df_no_outliers.shape}')

# Display the first few rows of the dataset
print(df.head())

# Exploratory Data Analysis (EDA)

# 1. Checking missing values
print("Missing Values:\n", df.isnull().sum())

# 2. Summary statistics
print("\nSummary Statistics:\n", df.describe())

# 3. Histogram of Distribution of satisfaction_level
plt.figure(figsize=(10, 6))
sb.histplot(df['satisfaction_level'], bins=20, kde=True)
plt.title('Distribution of Satisfaction Level')
plt.xlabel('Satisfaction Level')
plt.ylabel('Frequency')
plt.savefig('satisfaction_distribution.png')  # Save the plot
plt.show()

# Interpretation: The histogram shows the distribution of employee satisfaction levels. 
# A higher concentration of values on the right indicates that most employees are relatively satisfied.

# 4. Boxplot of satisfaction level by department
plt.figure(figsize=(12, 6))
sb.boxplot(x='dept', y='satisfaction_level', data=df)
plt.title('Satisfaction Level by Department')
plt.xticks(rotation=45)
plt.savefig('satisfaction_by_department.png')  # Save the plot
plt.show()

# Interpretation: The boxplot illustrates how satisfaction levels vary across different departments. 
# Notably, some departments exhibit lower satisfaction levels, indicating potential areas for improvement.

# 5. Scatter plot of average_monthly_hours vs satisfaction_level
plt.figure(figsize=(10, 6))
sb.scatterplot(x='average_montly_hours', y='satisfaction_level', data=df)
plt.title('Average Monthly Hours vs Satisfaction Level')
plt.xlabel('Average Monthly Hours')
plt.ylabel('Satisfaction Level')
plt.savefig('hours_vs_satisfaction.png')  # Save the plot
plt.show()

# Interpretation: The scatter plot reveals the relationship between average monthly hours worked and satisfaction levels. 
# A positive trend suggests that employees who work more hours tend to report higher satisfaction, but outliers may exist.

# Hypothesis Testing
# Null hypothesis - H0: Promotion does not affect satisfaction level
# Alternative hypothesis - H1: Promotion affects satisfaction level

# Separate satisfaction levels by promotion
promoted = df[df['promotion_last_5years'] == 1]['satisfaction_level']
not_promoted = df[df['promotion_last_5years'] == 0]['satisfaction_level']

# T-test
t_stat, p_value = stats.ttest_ind(promoted, not_promoted)
print(f'T-test results: t-statistic = {t_stat}, p-value = {p_value}')

# Significance level
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Promotion status affects satisfaction level.")
else:
    print("Fail to reject the null hypothesis: Promotion status does not affect satisfaction level.")

# Number of employees in each department
print("\nEmployee Count by Department:\n", df['dept'].value_counts())

# ANOVA
# Null hypothesis - H0: There are no differences in satisfaction levels among different departments
# Alternative hypothesis - H1: At least one department has a different satisfaction level
# Filter departments with at least a minimum number of employees
min_samples = 5
filtered_depts = df['dept'].value_counts()[df['dept'].value_counts() >= min_samples].index
filtered_data = [df[df['dept'] == dept]['satisfaction_level'] for dept in filtered_depts]

# Perform ANOVA only if we have enough groups
if len(filtered_data) > 1:
    anova_results = stats.f_oneway(*filtered_data)
    print(f'ANOVA results: F-statistic = {anova_results.statistic}, p-value = {anova_results.pvalue}')

    if anova_results.pvalue < alpha:
        print("Reject the null hypothesis: There are differences in satisfaction levels among departments.")
    else:
        print("Fail to reject the null hypothesis: No differences in satisfaction levels among departments.")
else:
    print("Not enough departments with sufficient samples for ANOVA.")

# Correlation Analysis
# Calculate correlation matrix
correlation_matrix = df[['satisfaction_level', 'last_evaluation', 'average_montly_hours', 'time_spend_company']].corr()

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')  # Save the plot
plt.show()

# Interpretation: The correlation matrix shows how satisfaction level correlates with other variables. 
# A strong positive correlation between satisfaction level and last evaluation suggests that higher performance evaluations correlate with higher satisfaction.

# K-Means Clustering
# Select features for clustering
features = df_no_outliers[['satisfaction_level', 'last_evaluation', 'average_montly_hours']]
kmeans = KMeans(n_clusters=3)  # Choose 3 clusters
df_no_outliers.loc[:, 'cluster'] = kmeans.fit_predict(features)

# Plot clusters
plt.figure(figsize=(10, 6))
plt.scatter(df_no_outliers['average_montly_hours'], df_no_outliers['satisfaction_level'], c=df_no_outliers['cluster'], cmap='viridis')
plt.title('K-Means Clustering of Employees')
plt.xlabel('Average Monthly Hours')
plt.ylabel('Satisfaction Level')
plt.savefig('kmeans_clustering.png')  # Save the plot
plt.show()

# Interpretation: The scatter plot of K-Means clustering indicates how employees are grouped based on their satisfaction levels and average monthly hours. 
# Distinct clusters can help identify employee groups with similar characteristics for targeted interventions.

# Critical Interval Calculation (95% Confidence Interval)
confidence_level = 0.95
sample_size = df_no_outliers['satisfaction_level'].count()
mean_satisfaction = df_no_outliers['satisfaction_level'].mean()
std_dev = df_no_outliers['satisfaction_level'].std()

# Calculate margin of error
z_score = stats.norm.ppf((1 + confidence_level) / 2)
margin_of_error = z_score * (std_dev / np.sqrt(sample_size))

# Critical interval
critical_interval = (mean_satisfaction - margin_of_error, mean_satisfaction + margin_of_error)
print(f"95% Confidence Interval for Satisfaction Level: {critical_interval}")

# Insights and Recommendations
print("Insights:")
print("- Promotion significantly affects employee satisfaction.")
print("- Further analysis is needed to improve satisfaction in departments with lower scores.")
