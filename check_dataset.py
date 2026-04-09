import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("fire_dataset.csv")

# 2. Try to detect target column automatically
possible_targets = ['label', 'fire', 'class', 'target', 'status']
target_col = None
for col in df.columns:
    if col.lower() in possible_targets:
        target_col = col
        break

if target_col is None:
    print("⚠️ Could not find a target column automatically.")
    print("Available columns:", df.columns)
    exit()

print(f"✅ Using target column: {target_col}")

# 3. Check class balance
print("\nClass distribution:")
print(df[target_col].value_counts())

# 4. Inspect feature ranges
features_to_check = [c for c in df.columns if c.lower() in ['brightness','confidence','frp']]
if features_to_check:
    print("\nFeature ranges by class:")
    print(df.groupby(target_col)[features_to_check].describe())
else:
    print("\n⚠️ Could not find Brightness/Confidence/FRP columns.")

# 5. Check for missing/invalid values
print("\nMissing values per column:")
print(df.isnull().sum())

# 6. Quick feature importance (Random Forest)
X = df.drop(target_col, axis=1)
y = df[target_col]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importances = model.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(8,6))
plt.barh(feature_names, importances, color="orange")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Fire Detection")
plt.show()