"""
Data Leakage Detection Script
Check for suspicious perfect correlations with target variable
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
DATA_PATH = r"e:\hafiza mam work\BDHS2022_MH_ML_ready_1.csv"
df_raw = pd.read_csv(DATA_PATH, dtype=str)
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.replace(r'^\s*$', np.nan, regex=True)
df = df_raw.apply(pd.to_numeric, errors='coerce')

if 'CASEID' in df.columns:
    df = df.drop(columns=['CASEID'])

print("="*70)
print("DATA LEAKAGE DETECTION ANALYSIS")
print("="*70)

# Basic info
print(f"\n[1] Dataset Shape: {df.shape}")
print(f"    Target variable: dep")
print(f"    Depression prevalence: {df['dep'].mean()*100:.2f}%")

# Check for perfect correlations
print("\n[2] Checking correlations with target (dep)...")
DROP_COLS = ['anx', 'disu', 'dep']
features = [c for c in df.columns if c not in DROP_COLS]

correlations = []
for col in features:
    if df[col].isnull().all():
        continue
    corr = df[[col, 'dep']].corr().iloc[0, 1]
    if not np.isnan(corr):
        correlations.append({'Feature': col, 'Correlation': abs(corr)})

corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
print("\n   Top 15 Correlations with Depression:")
print(corr_df.head(15).to_string(index=False))

# Check for suspiciously high correlations
suspicious = corr_df[corr_df['Correlation'] > 0.8]
if len(suspicious) > 0:
    print(f"\n   ⚠️  ALERT: {len(suspicious)} features with correlation > 0.8 (DATA LEAKAGE RISK!)")
    print(suspicious.to_string(index=False))
else:
    print("\n   ✅ No suspiciously high correlations (> 0.8)")

# Check for duplicate rows
print(f"\n[3] Checking for duplicate rows...")
n_duplicates = df.duplicated().sum()
print(f"    Duplicate rows: {n_duplicates}")
if n_duplicates > 100:
    print(f"    ⚠️  WARNING: {n_duplicates} duplicates found!")

# Check unique values in target
print(f"\n[4] Target variable distribution:")
print(df['dep'].value_counts().to_string())

# Check if any feature has perfect separation
print(f"\n[5] Checking for perfect separators...")
df_clean = df.dropna()
for col in features[:20]:  # Check first 20 features
    if col not in df_clean.columns:
        continue
    # For each unique value, check if it predicts depression perfectly
    for val in df_clean[col].unique()[:10]:  # Check first 10 unique values
        subset = df_clean[df_clean[col] == val]
        if len(subset) > 10:  # Only if enough samples
            dep_rate = subset['dep'].mean()
            if dep_rate == 0 or dep_rate == 1:
                print(f"    ⚠️  {col} = {val}: {len(subset)} samples, {dep_rate*100:.0f}% depressed (PERFECT SEPARATOR!)")

# Check selected features from previous run
selected_features = ['MTH22', 'MTH24', 'V005', 'V012', 'V021', 'V023', 'V025', 
                     'V212', 'V312', 'V511', 'V632', 'V701', 'V730',
                     'age_cat', 'age_dif', 'age_dif_cat', 'age_fb_cat', 
                     'contra', 'contra_decision3']

print(f"\n[6] Analyzing SELECTED features for leakage...")
for feat in selected_features:
    if feat in corr_df['Feature'].values:
        corr_val = corr_df[corr_df['Feature'] == feat]['Correlation'].values[0]
        if corr_val > 0.5:
            print(f"    {feat:20s}: correlation = {corr_val:.3f} {'⚠️  HIGH!' if corr_val > 0.7 else ''}")

# Test simple model on single best feature
print(f"\n[7] Testing single-feature prediction...")
best_feature = corr_df.iloc[0]['Feature']
print(f"    Using best feature: {best_feature}")

df_test = df[[best_feature, 'dep']].dropna()
X = df_test[[best_feature]].values
y = df_test['dep'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42, stratify=y)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, dt.predict(X_test))
print(f"    Single-feature Decision Tree accuracy: {acc*100:.2f}%")
if acc > 0.95:
    print(f"    ⚠️  ALERT: Single feature achieves {acc*100:.0f}% accuracy - likely DATA LEAKAGE!")

print("\n" + "="*70)
print("RECOMMENDATION:")
if len(suspicious) > 0 or acc > 0.95:
    print("⚠️  DATA LEAKAGE DETECTED!")
    print("   Remove highly correlated features or investigate data quality")
else:
    print("✅ No obvious data leakage detected")
    print("   100% accuracy might be due to:")
    print("   - Very strong predictive patterns in the data")
    print("   - Small test set with clear separability")
    print("   - Need to verify with external validation set")
print("="*70)
