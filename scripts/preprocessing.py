import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import pickle


def preprocess_and_split(df):
    # Print the shape of the DataFrame
    print("Shape of the DataFrame:",df.shape)
    # Print the columns of the DataFrame
    print("Columns of the DataFrame:",df.columns)
    #handeling missing values
    if df.isnull().sum().sum() > 0:
        print("Missing values found:")
        print(df.isnull().sum())
        df = df.dropna()
    else:
        print("No missing values found.")

    # handeling repeated rows
    if df.duplicated().sum() > 0:
        print("Repeated rows found:")
        print(df[df.duplicated()])
        df = df.drop_duplicates()
    else:
        print("No repeated rows found.")
    
    # encode categorical variable
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        print(f"Categorical columns {categorical_columns} encoded using one-hot encoding.")
    else:
        print("No categorical columns found.")

    if 'HeartDisease' not in df.columns:
        raise ValueError("Target column 'HeartDisease' not found in the DataFrame.")

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    # 📊 Identify numeric columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = X.columns.difference(numeric_cols)

    # 📏 Apply scaling only on numeric columns
    scaler = StandardScaler()
    X_scaled_numeric = scaler.fit_transform(X[numeric_cols])

    # Convert scaled numeric features back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled_numeric, columns=numeric_cols, index=X.index)

    # 🧱 Combine scaled numeric and unscaled non-numeric features
    X_processed = pd.concat([X_scaled_df, X[non_numeric_cols]], axis=1)

    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_processed, y)

    selector = SelectFromModel(rf, threshold='median', prefit=True)
    X_selected = selector.transform(X_processed)

    original_features = X_processed.columns.tolist()
    selected_features = np.array(original_features)[selector.get_support()].tolist()
    print(f"🎯 Selected {len(selected_features)} important features out of {len(original_features)}")

    # 💾 Save selected features
    with open('models/selected_features.txt', 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    # ✂️ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, selected_features, scaler


