#!/usr/bin/env python3
"""
US Census Data Cleaning and Analysis Portfolio
Author: Data Analyst Candidate
Date: Current Date

This script demonstrates professional data cleaning and visualization skills
using US Census data. It processes multiple CSV files, cleans demographic data,
and creates insightful visualizations.
"""

# =============================================================================
# 1) IMPORTS AND CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns

# Set display options for clean output formatting
pd.options.display.float_format = "{:,.2f}".format
plt.style.use('seaborn-v0_8')

# =============================================================================
# 2) DATA ACQUISITION - READING MULTIPLE CSV FILES
# =============================================================================

# Identify all census data files matching the pattern
print("🔍 Searching for census data files...")
files = glob.glob('states*.csv')
print(f"📁 Found {len(files)} files: {files}")

# Read and combine all CSV files into a single DataFrame
df_list = []
for file in files:
    print(f"📊 Loading data from {file}...")
    data = pd.read_csv(file)
    df_list.append(data)

# Concatenate all DataFrames into one comprehensive dataset
df = pd.concat(df_list, ignore_index=True)
print(f"✅ Combined dataset shape: {df.shape}")

# =============================================================================
# 3) DATA CLEANING - INCOME COLUMN
# =============================================================================

print("\n💰 Cleaning Income column...")
# Remove dollar signs and commas, convert to numeric format
df['Income'] = df['Income'].replace('[$,]', '', regex=True).astype(float)
print(f"📈 Income statistics: Mean=${df['Income'].mean():,.0f}, Max=${df['Income'].max():,.0f}")

# =============================================================================
# 4) DATA CLEANING - GENDER POPULATION SPLITTING
# =============================================================================

print("\n👥 Processing gender population data...")
# Split GenderPop column into separate Male and Female columns
gender_split = df['GenderPop'].str.split('_', expand=True)

# Remove trailing 'M' and 'F' characters, convert to integers
df[['Male', 'Female']] = gender_split.apply(lambda x: x.str[:-1]).replace('', pd.NA).astype('Int64')

# Remove original GenderPop column and reorder columns
df = df[['State', 'TotalPop', 'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific', 'Income', 'Male', 'Female']]
print("✅ Gender data successfully split and formatted")

# =============================================================================
# 5) DATA CLEANING - PERCENTAGE COLUMNS
# =============================================================================

print("\n📊 Cleaning percentage-based demographic columns...")
# List of demographic columns containing percentage data
demographic_cols = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']

# Remove percentage signs and convert to float values
df[demographic_cols] = df[demographic_cols].replace('%', '', regex=True).astype(float)
print("✅ Percentage formatting removed from demographic data")

# =============================================================================
# 6) DATA QUALITY - DUPLICATE HANDLING
# =============================================================================

print("\n🔍 Checking for duplicate entries...")
initial_count = len(df)
duplicate_count = df.duplicated().sum()

print(f"📈 Found {duplicate_count} duplicate rows out of {initial_count} total rows")

# Remove duplicates and reset index for clean data structure
df = df.drop_duplicates().reset_index(drop=True)
final_count = len(df)
print(f"✅ Removed {initial_count - final_count} duplicates. Final dataset: {final_count} rows")

# =============================================================================
# 7) FEATURE ENGINEERING - FEMALE PROPORTION
# =============================================================================

print("\n🎛️ Creating derived features...")
# Calculate proportion of female population for each state
df['Female_Proportion'] = df['Female'] / df['TotalPop']
df['Female_Proportion'] = df['Female_Proportion'].astype(float)
print("✅ Female proportion feature created")

# =============================================================================
# 8) VISUALIZATION 1 - INCOME VS FEMALE PROPORTION
# =============================================================================

print("\n📈 Creating Income vs Female Proportion visualization...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Female_Proportion', y='Income', hue='State', s=80, alpha=0.7)

plt.xlabel('Proportion of Female Population', fontsize=12)
plt.ylabel('Average Income ($)', fontsize=12)
plt.title('State Income vs Female Population Proportion', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add trend line to show correlation
z = np.polyfit(df['Female_Proportion'], df['Income'], 1)
p = np.poly1d(z)
plt.plot(df['Female_Proportion'], p(df['Female_Proportion']), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig('income_vs_female_proportion.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Scatter plot saved as 'income_vs_female_proportion.png'")

# =============================================================================
# 9) DATA QUALITY CHECK - MISSING VALUES
# =============================================================================

print("\n🔍 Assessing data completeness...")
print("Missing values per column:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# =============================================================================
# 10) DATA CLEANING - HANDLING MISSING DEMOGRAPHIC VALUES
# =============================================================================

print("\n🔄 Imputing missing demographic values...")
# Fill missing demographic percentages by distributing remaining percentage
df[demographic_cols] = df[demographic_cols].apply(
    lambda row: row.fillna(100 - row.sum()) if row.isna().any() else row, 
    axis=1
)
print("✅ Missing demographic values imputed")

# =============================================================================
# 11) VISUALIZATION 2 - DEMOGRAPHIC DISTRIBUTION HISTOGRAMS
# =============================================================================

print("\n📊 Creating demographic distribution histograms...")
# Create individual histograms for each demographic group
for demographic in demographic_cols:
    plt.figure(figsize=(10, 6))
    plt.hist(df[demographic], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    
    plt.xlabel(f'{demographic} Population Percentage', fontsize=12)
    plt.ylabel('Number of States', fontsize=12)
    plt.title(f'Distribution of {demographic} Population Across States', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistical annotations
    mean_val = df[demographic].mean()
    median_val = df[demographic].median()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}%')
    plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}%')
    plt.legend()
    
    filename = f'{demographic.lower()}_distribution.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Histogram saved as '{filename}'")

# =============================================================================
# 12) DATASET SUMMARY AND VALIDATION
# =============================================================================

print("\n" + "="*50)
print("📋 FINAL DATASET SUMMARY")
print("="*50)

print(f"📊 Dataset Shape: {df.shape}")
print(f"🏛️ Number of States: {df['State'].nunique()}")
print(f"👥 Total Population Represented: {df['TotalPop'].sum():,}")

print("\n📈 Key Statistics:")
print(df[['TotalPop', 'Income', 'Female_Proportion'] + demographic_cols].describe())

print("\n🏆 Top 5 States by Income:")
top_income_states = df.nlargest(5, 'Income')[['State', 'Income', 'Female_Proportion']]
print(top_income_states.to_string(index=False))

print("\n✅ Data cleaning and analysis completed successfully!")
print("📁 Output files generated:")
print("   - income_vs_female_proportion.png")
for demo in demographic_cols:
    print(f"   - {demo.lower()}_distribution.png")

# =============================================================================
# 13) DATA EXPORT (OPTIONAL)
# =============================================================================

# Export cleaned data for future use
df.to_csv('cleaned_us_census_data.csv', index=False)
print("\n💾 Cleaned dataset exported as 'cleaned_us_census_data.csv'")