# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Data
df = pd.read_csv('your_data.csv')   # <-- Change this to your data file

# Step 3: Data Exploration (EDA)
print("Shape of data:", df.shape)
print("Columns:", df.columns)
print(df.head())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# Step 4: Data Cleaning
# Fill missing numerical values with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates()

# Step 5: Feature Engineering
# Example: Create a new feature (customize as needed)
# df['ratio'] = df['feature1'] / (df['feature2'] + 1)

# Step 6: Data Visualization
plt.figure(figsize=(10,5))
sns.histplot(df.select_dtypes(include=np.number).iloc[:,0], kde=True)
plt.title('Distribution of First Numeric Column')
plt.show()

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 7: Statistical Analysis
group_col = df.columns[0]
if df[group_col].dtype == 'object':
    # Example: Groupby and aggregate
    grouped = df.groupby(group_col).mean()
    print(f"\nMean values grouped by {group_col}:\n", grouped)

# Step 8: Simple Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

numeric_cols = df.select_dtypes(include=np.number).columns
if len(numeric_cols) > 1:
    X = df[numeric_cols[1:]]
    y = df[numeric_cols[0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Test set score:", model.score(X_test, y_test))

# Step 9: Conclusion
print("Data analysis complete.")