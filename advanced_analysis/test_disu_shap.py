import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

DATA_PATH = r"e:\hafiza mam work\BDHS2022_MH_ML_ready_1.csv"
df = pd.read_csv(DATA_PATH).apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['disu'].astype(int)
X = df.drop(columns=['dep', 'anx', 'disu', 'v005', 'v021', 'v023', 'MTH22', 'MTH24', 'CASEID'], errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_test.head(100), check_additivity=False)
print("SV type:", type(sv))
if isinstance(sv, list):
    print("SV list length:", len(sv))
    print("SV[0] shape:", sv[0].shape)
else:
    print("SV shape:", sv.shape)
