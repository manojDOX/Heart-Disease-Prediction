from data_loader import load_data
from preprocessing import preprocess_and_split
from train_models import train_models
from evaluate_models import evaluate_models

if __name__ == "__main__":
    try:
        df = load_data()
        print("\n📊 First 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"you have issue in loading data: {e}")
    X_train, X_test, y_train, y_test, features, scaler = preprocess_and_split(df)

    print(f"\n✅ Final Split:")
    print(f"🔹 Training Samples: {X_train.shape[0]}")
    print(f"🔹 Testing Samples: {X_test.shape[0]}")
    print(f"📌 Features Used: {features}")

    print("🏋️ Training models...")
    train_models(X_train, y_train)

    print("📈 Evaluating models...")
    results = evaluate_models(X_test, y_test)

    print("\n✅ Pipeline Complete!")