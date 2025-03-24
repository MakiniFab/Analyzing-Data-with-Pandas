import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
species_map = {i: species for i, species in enumerate(iris.target_names)}
df['species'] = df['species'].map(species_map)

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# No missing values, but handling any potential issues
# Fill missing values if any (not applicable here, but general approach)
df.fillna(df.mean(), inplace=True)

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Grouping by species and computing mean of numerical columns
species_mean = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(species_mean)

# Task 3: Data Visualization
plt.figure(figsize=(12, 8))

# Line Chart: Sepal Length Trend (Example time series with index as pseudo-time)
plt.subplot(2, 2, 1)
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.title('Trend of Sepal Length over Observations')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()

# Bar Chart: Average Sepal Width per Species
plt.subplot(2, 2, 2)
sns.barplot(x='species', y='sepal width (cm)', data=df, palette='viridis')
plt.title('Average Sepal Width by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Width (cm)')

# Histogram: Distribution of Petal Length
plt.subplot(2, 2, 3)
sns.histplot(df['petal length (cm)'], bins=20, kde=True, color='green')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

# Scatter Plot: Sepal Length vs. Petal Length
plt.subplot(2, 2, 4)
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='coolwarm')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')

plt.tight_layout()
plt.show()

