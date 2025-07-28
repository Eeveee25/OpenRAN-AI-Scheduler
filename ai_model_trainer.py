import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model():
    df = pd.read_csv('processed_network_data.csv')

    # --- NEW: Use the new standard deviation features ---
    features = ['avg_cqi', 'std_cqi', 'avg_rsrp', 'std_rsrp', 'num_ues']
    X = df[features]

    bins = [0, 0.2, 0.5, 1.1]
    labels = [0, 1, 2] 
    df['allocation_class'] = pd.cut(df['throughput_share'], bins=bins, labels=labels, right=False)
    df.dropna(subset=['allocation_class'], inplace=True)
    y = df['allocation_class']
    X = df.loc[y.index, features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Added stratify

    print("Training Random Forest model with new features...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("\nEvaluating new model on test set...")
    y_pred = model.predict(X_test)
    print(f"New Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("New Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

    joblib.dump(model, 'rf_allocator_model.pkl')
    print("\nModel saved to rf_allocator_model.pkl")

if __name__ == '__main__':
    train_model()