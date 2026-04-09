import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("fire_dataset.csv")

# 2. Create binary target column (1 = Fire, 0 = No Fire)
df['fire_label'] = (df['type'] == 1).astype(int)

# 3. Select features
features = ['latitude','longitude','brightness','scan','track',
            'confidence','bright_t31','frp']
X = df[features]
y = df['fire_label']

# 4. Balance dataset (simple undersampling of No Fire)
fire_df = df[df['fire_label'] == 1]
no_fire_df = df[df['fire_label'] == 0].sample(len(fire_df), random_state=42)
balanced_df = pd.concat([fire_df, no_fire_df])

X = balanced_df[features]
y = balanced_df['fire_label']

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")