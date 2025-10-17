| Dataset         | Best Model           | Why                                                                 |
|-----------------|----------------------|---------------------------------------------------------------------|
| Telco Churn     | Logistic Regression  | Handles mixed data types best; semi-supervised approach works well. |
| UCI Spambase    | Local Outlier Factor | All numerical; excels at finding local density-based anomalies.     |
| Pima Diabetes   | K-Means              | All numerical; good for finding cluster-based or distance anomalies.|
| UCI Air Quality | Isolation Forest     | Effective for time series data; detects anomalies effectively. |