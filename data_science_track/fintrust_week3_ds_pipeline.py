from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data() -> pd.DataFrame:
    customer = pd.read_csv(ROOT / "customer.csv")
    transactions = pd.read_csv(ROOT / "transactions.csv")
    transactions["Transaction_DateTime"] = pd.to_datetime(
        transactions["Transaction_DateTime"], errors="coerce"
    )
    transactions = transactions.sort_values("Transaction_DateTime").reset_index(drop=True)

    merged = transactions.merge(customer, on="Customer_ID", how="left", validate="many_to_one")
    merged = merged.sort_values(["Customer_ID", "Transaction_DateTime"]).reset_index(drop=True)

    merged["log_amount"] = np.log1p(merged["Amount_NGN"])
    merged["transaction_hour"] = merged["Transaction_DateTime"].dt.hour
    merged["day_of_week"] = merged["Transaction_DateTime"].dt.dayofweek
    merged["is_weekend"] = (merged["Transaction_DateTime"].dt.dayofweek >= 5).astype(int)
    merged["is_night"] = merged["transaction_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    merged["prev_tx_amount"] = merged.groupby("Customer_ID")["Amount_NGN"].transform(
        lambda s: s.shift(1)
    )
    merged["time_since_last_tx_hours"] = (
        merged.groupby("Customer_ID")["Transaction_DateTime"]
        .diff()
        .dt.total_seconds()
        .div(3600)
        .fillna(0)
    )
    merged["customer_tx_count"] = merged.groupby("Customer_ID").cumcount() + 1
    merged["amount_vs_customer_median"] = (
        merged["Amount_NGN"] - merged.groupby("Customer_ID")["Amount_NGN"].transform("median")
    )
    merged["amount_vs_customer_mean"] = (
        merged["Amount_NGN"] - merged.groupby("Customer_ID")["Amount_NGN"].transform("mean")
    )

    merged["Risk_Review_Flag"] = merged["Risk_Review_Flag"].map({"No": 0, "Yes": 1})
    return merged


def save_model_comparison(results: list[dict]) -> pd.DataFrame:
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values("f1", ascending=False).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    return comparison


def save_confusion_matrix(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=200)
    plt.close(fig)


def save_feature_importance(pipe: Pipeline, model_name: str) -> None:
    model = pipe.named_steps["model"]
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "coef_"):
        importance = model.coef_[0]
        feature_df = pd.DataFrame({"feature": feature_names, "coefficient": importance})
        feature_df["abs_coefficient"] = feature_df["coefficient"].abs()
        feature_df = feature_df.sort_values("abs_coefficient", ascending=False).drop(columns=["abs_coefficient"])
        output_path = OUT_DIR / "logistic_regression_coefficients.csv"
        feature_df.to_csv(output_path, index=False)


def create_error_analysis_report(full_test_df: pd.DataFrame, y_pred: np.ndarray, probas: np.ndarray) -> pd.DataFrame:
    report = full_test_df.copy()
    report["actual"] = report["Risk_Review_Flag"].astype(int)
    report["predicted"] = y_pred.astype(int)
    report["probability"] = probas
    report["false_positive"] = (report["actual"] == 0) & (report["predicted"] == 1)
    report["false_negative"] = (report["actual"] == 1) & (report["predicted"] == 0)

    summary_parts = []
    for pattern_name, flag_name in [("false_positive", "false_positive"), ("false_negative", "false_negative")]:
        subset = report[report[flag_name]]
        if subset.empty:
            continue
        grouped = subset.groupby(["Transaction_Type", "Channel"]).size().reset_index(name="count")
        grouped["error_type"] = pattern_name
        summary_parts.append(grouped)

    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True)
    else:
        summary = pd.DataFrame(columns=["Transaction_Type", "Channel", "count", "error_type"])
    summary.to_csv(OUT_DIR / "error_analysis_summary.csv", index=False)
    return summary


def main() -> None:
    data = load_and_prepare_data()
    target = "Risk_Review_Flag"
    leakage_cols = [
        "Transaction_ID",
        "Customer_ID",
        "Customer_Name",
        "Transaction_Status",
        "Account_Status",
        "Transaction_DateTime",
    ]

    feature_frame = data.drop(columns=leakage_cols + [target], errors="ignore").copy()
    target_series = data[target].copy()

    split_date = data["Transaction_DateTime"].quantile(0.8)
    train_mask = data["Transaction_DateTime"] <= split_date

    X_train = feature_frame.loc[train_mask].copy()
    X_test = feature_frame.loc[~train_mask].copy()
    y_train = target_series.loc[train_mask].copy()
    y_test = target_series.loc[~train_mask].copy()

    num_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", C=0.7),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            max_depth=12,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
    }

    all_results = []
    selected_model_name = None
    selected_pipeline = None
    selected_proba = None
    selected_pred = None
    selected_full_test = None
    best_score = -1.0

    for model_name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
        }
        all_results.append(metrics)

        candidate_score = metrics["roc_auc"] + 0.5 * metrics["recall"]
        if candidate_score > best_score:
            selected_model_name = model_name
            selected_pipeline = pipeline
            selected_proba = proba
            selected_pred = pred
            selected_full_test = data.loc[~train_mask].copy()
            best_score = candidate_score

        save_confusion_matrix(model_name, y_test, pred)

    comparison = save_model_comparison(all_results)
    save_feature_importance(selected_pipeline, selected_model_name)
    create_error_analysis_report(selected_full_test, selected_pred, selected_proba)

    final_model_path = ROOT / "candidate_model.pkl"
    with open(final_model_path, "wb") as fh:
        pickle.dump(selected_pipeline, fh)

    final_summary = {
        "selected_model": selected_model_name,
        "time_split_date": str(split_date.date()),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "comparison": comparison.to_dict(orient="records"),
    }
    with open(OUT_DIR / "final_model_summary.json", "w", encoding="utf-8") as fh:
        json.dump(final_summary, fh, indent=2)

    prepared_dataset = data.drop(columns=["Transaction_ID", "Customer_ID", "Customer_Name", "Transaction_Status", "Account_Status"], errors="ignore").copy()
    prepared_dataset.to_csv(ROOT / "prepared_modelling_dataset.csv", index=False)

    print("Model comparison saved to:", OUT_DIR / "model_comparison.csv")
    print("Selected model:", selected_model_name)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
