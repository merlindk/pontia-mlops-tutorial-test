import pandas as pd
import logging
import time
from pathlib import Path
import platform
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

# Configurar logging (consola + archivo)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# MLflow config
MLFLOW_URI = "http://57.151.65.76:5000"  # Replace with your real URL
EXPERIMENT_NAME = "adult-income"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Column names
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

def load_data(train_path, test_path):
    logging.info("Loading training and test datasets...")
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS, na_values=" ?", skipinitialspace=True)
    test_df = pd.read_csv(test_path, header=0, names=COLUMNS, na_values=" ?", skipinitialspace=True, skiprows=1)
    test_df["income"] = test_df["income"].str.replace(".", "", regex=False)

    logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    logging.info(f"Missing values (train): {train_df.isnull().sum().sum()}, (test): {test_df.isnull().sum().sum()}")
    return train_df.dropna(), test_df.dropna()

def preprocess_data(train_df, test_df):
    logging.info("Preprocessing and encoding features...")
    cat_cols = train_df.select_dtypes(include="object").columns.drop("income")
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le

    train_df["income"] = train_df["income"].apply(lambda x: 1 if x == ">50K" else 0)
    test_df["income"] = test_df["income"].apply(lambda x: 1 if x == ">50K" else 0)

    X_train = train_df.drop("income", axis=1)
    y_train = train_df["income"]
    X_test = test_df.drop("income", axis=1)
    y_test = test_df["income"]

    scaler = StandardScaler()
    logging.info("Scaling features...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logging.info(f"Feature columns: {list(X_train.columns)}")
    logging.info(f"X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), scaler, label_encoders

def train_model(X_train, y_train):
    logging.info("Training RandomForestClassifier...")
    model = RandomForestClassifier(random_state=42)
    logging.info(f"Model parameters: {model.get_params()}")
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    logging.info("Evaluating model...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Classification Report:\n{report}")
    return acc, report

def main():
    script_start = time.time()
    logging.info(f"System info: {platform.platform()}")

    train_df, test_df = load_data(DATA_DIR / "adult.data", DATA_DIR / "adult.test")
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(train_df, test_df)
    mlflow.autolog()
    with mlflow.start_run(run_name=run_name):
        start_time = time.time()
        model = train_model(X_train, y_train)
        elapsed = time.time() - start_time
        logging.info(f"Model training complete. Time taken: {elapsed:.2f} seconds")
        evaluate(model, X_test, y_test)


        # Save and log model with metadata
        model_path = MODEL_DIR / "model.pkl"
        joblib.dump(model, model_path)

        # Save and log scaler and encoders
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(encoders, MODEL_DIR / "encoders.pkl")

    total_time = time.time() - script_start
    logging.info(f"Script completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
