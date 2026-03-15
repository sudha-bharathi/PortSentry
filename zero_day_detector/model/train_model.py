from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "MachineLearningCVE"
MODEL_DIR = BASE_DIR / "saved_model"
MODEL_PATH = MODEL_DIR / "anomaly_model.pkl"

MODEL_DIR.mkdir(exist_ok=True)


def safe_numeric_column(df: pd.DataFrame, col_name: str, default=0) -> pd.Series:
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def load_all_csvs(dataset_dir: Path) -> pd.DataFrame:
    csv_files = list(dataset_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")

    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            dataframes.append(df)
            print(f"Loaded: {file.name} -> {df.shape}")
        except Exception as e:
            print(f"Skipped {file.name}: {e}")

    if not dataframes:
        raise ValueError("No CSV files could be loaded.")

    combined = pd.concat(dataframes, ignore_index=True)
    print(f"Combined shape: {combined.shape}")
    return combined


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_columns(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how="all", inplace=True)

    if "label" not in df.columns:
        raise ValueError("Could not find 'label' column in dataset.")

    # Packet totals
    df["packets"] = pd.Series(0, index=df.index, dtype="float64")
    if "total_fwd_packets" in df.columns:
        df["packets"] += safe_numeric_column(df, "total_fwd_packets", 0)
    if "total_backward_packets" in df.columns:
        df["packets"] += safe_numeric_column(df, "total_backward_packets", 0)

    # Byte totals
    df["bytes"] = pd.Series(0, index=df.index, dtype="float64")
    if "total_length_of_fwd_packets" in df.columns:
        df["bytes"] += safe_numeric_column(df, "total_length_of_fwd_packets", 0)
    if "total_length_of_bwd_packets" in df.columns:
        df["bytes"] += safe_numeric_column(df, "total_length_of_bwd_packets", 0)

    # Duration in seconds
    df["duration"] = safe_numeric_column(df, "flow_duration", 0) / 1_000_000.0
    df["duration"] = df["duration"].clip(lower=0.001)

    # Rates
    df["pps"] = safe_numeric_column(df, "flow_packets_s", 0)
    df["bps"] = safe_numeric_column(df, "flow_bytes_s", 0)

    # Average packet size
    df["avg_packet_size"] = df["bytes"] / df["packets"].replace(0, 1)

    # Ports
    df["src_port"] = pd.Series(0, index=df.index, dtype="float64")
    df["dst_port"] = safe_numeric_column(df, "destination_port", 0)

    # Protocol
    protocol_series = safe_numeric_column(df, "protocol", 0)
    df["is_tcp"] = (protocol_series == 6).astype(int)
    df["is_udp"] = (protocol_series == 17).astype(int)

    # Sensitive destination ports
    sensitive_ports = {21, 22, 23, 53, 80, 135, 137, 138, 139, 443, 445, 1433, 3306, 3389}
    df["is_sensitive_port"] = df["dst_port"].isin(sensitive_ports).astype(int)

    # TCP flag summary
    flag_cols = [c for c in ["syn_flag_count", "ack_flag_count", "psh_flag_count", "urg_flag_count"] if c in df.columns]
    if flag_cols:
        df["tcp_flag_total"] = pd.Series(0, index=df.index, dtype="float64")
        for c in flag_cols:
            df["tcp_flag_total"] += safe_numeric_column(df, c, 0)
    else:
        df["tcp_flag_total"] = pd.Series(0, index=df.index, dtype="float64")

    feature_columns = [
        "duration",
        "packets",
        "bytes",
        "pps",
        "bps",
        "avg_packet_size",
        "src_port",
        "dst_port",
        "is_tcp",
        "is_udp",
        "is_sensitive_port",
        "tcp_flag_total",
    ]

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["label"].astype(str).str.strip().str.lower()

    return X, y


def train_and_save_model(X: pd.DataFrame, y: pd.Series, model_path: Path) -> None:
    normal_df = X[y == "benign"]

    if normal_df.empty:
        raise ValueError("No benign rows found. The model should be trained on benign traffic for anomaly detection.")

    print(f"Training rows used: {normal_df.shape[0]}")
    print(f"Feature columns: {list(normal_df.columns)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=150,
            contamination=0.05,
            random_state=42
        ))
    ])

    pipeline.fit(normal_df)
    joblib.dump(pipeline, model_path)

    print(f"Model saved to: {model_path}")
    print("Training complete.")


def main():
    print(f"Reading dataset from: {DATASET_DIR}")
    raw_df = load_all_csvs(DATASET_DIR)
    X, y = build_features(raw_df)
    train_and_save_model(X, y, MODEL_PATH)


if __name__ == "__main__":
    main()