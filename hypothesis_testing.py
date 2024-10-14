# hypothesis_testing.py
import pandas as pd
from scipy import stats

def perform_hypothesis_testing(df):
    # T-test
    promoted = df[df['promotion_last_5years'] == 1]['satisfaction_level']
    not_promoted = df[df['promotion_last_5years'] == 0]['satisfaction_level']

    t_stat, p_value = stats.ttest_ind(promoted, not_promoted)
    print(f'T-test results: t-statistic = {t_stat}, p-value = {p_value}')

    # ANOVA
    min_samples = 5
    filtered_depts = df['dept'].value_counts()[df['dept'].value_counts() >= min_samples].index
    filtered_data = [df[df['dept'] == dept]['satisfaction_level'] for dept in filtered_depts]

    if len(filtered_data) > 1:
        anova_results = stats.f_oneway(*filtered_data)
        print(f'ANOVA results: F-statistic = {anova_results.statistic}, p-value = {anova_results.pvalue}')

if __name__ == "__main__":
    # Assuming cleaned_data is imported or passed here
    pass
