import os
import numpy as np
import pandas as pd


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[ETL] Successfully loaded raw data with shape: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data, handle missing values, and manage extreme outliers."""
    df_clean = df.copy()

    # Drop unnamed index columns if present from Kaggle exports
    unnamed_cols = [col for col in df_clean.columns if "Unnamed" in col]
    if unnamed_cols:
        df_clean.drop(columns=unnamed_cols, inplace=True)

    # Impute missing MonthlyIncome with median (standard banking practice)
    if "MonthlyIncome" in df_clean.columns:
        median_income = df_clean["MonthlyIncome"].median()
        df_clean["MonthlyIncome"] = df_clean["MonthlyIncome"].fillna(median_income)

    # Impute NumberOfDependents if present
    if "NumberOfDependents" in df_clean.columns:
        df_clean["NumberOfDependents"] = df_clean["NumberOfDependents"].fillna(0)

    # Cap extreme outliers in financial ratios (Winsorization)
    if "RevolvingUtilizationOfUnsecuredLines" in df_clean.columns:
        df_clean["RevolvingUtilizationOfUnsecuredLines"] = np.where(
            df_clean["RevolvingUtilizationOfUnsecuredLines"] > 5.0,
            5.0,
            df_clean["RevolvingUtilizationOfUnsecuredLines"]
        )

    if "DebtRatio" in df_clean.columns:
        df_clean["DebtRatio"] = np.where(
            df_clean["DebtRatio"] > 10.0,
            10.0,
            df_clean["DebtRatio"]
        )

    print("[ETL] Data cleaning and imputation complete.")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create financial risk indicators for Early Warning Signals (EWS)."""
    df_feat = df.copy()

    # 1. Total Delinquency Count across all past-due categories
    delinquency_cols = [
        col for col in df_feat.columns 
        if "PastDue" in col or "Late" in col
    ]
    if delinquency_cols:
        df_feat["TotalPastDueOccurrences"] = df_feat[delinquency_cols].sum(axis=1)

    # 2. High Credit Utilization Flag (> 80% utilization indicates liquidity stress)
    if "RevolvingUtilizationOfUnsecuredLines" in df_feat.columns:
        df_feat["IsHighUtilization"] = (
            df_feat["RevolvingUtilizationOfUnsecuredLines"] > 0.80
        ).astype(int)

    # 3. Monthly Debt Burden Estimate
    if "MonthlyIncome" in df_feat.columns and "DebtRatio" in df_feat.columns:
        df_feat["EstimatedMonthlyDebt"] = (
            df_feat["MonthlyIncome"] * df_feat["DebtRatio"]
        ).round(2)

    # 4. Income Per Dependent (if dependents exist)
    if "MonthlyIncome" in df_feat.columns and "NumberOfDependents" in df_feat.columns:
        df_feat["IncomePerHouseholdMember"] = (
            df_feat["MonthlyIncome"] / (df_feat["NumberOfDependents"] + 1)
        ).round(2)

    print(f"[ETL] Feature engineering complete. Total columns: {len(df_feat.columns)}")
    return df_feat


def run_pipeline(raw_path: str = "data/raw/credit_data.csv", output_path: str = "data/processed/cleaned_credit_data.csv") -> pd.DataFrame:
    """Execute full ETL pipeline and save processed output."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    raw_df = load_raw_data(raw_path)
    clean_df = preprocess_data(raw_df)
    processed_df = engineer_features(clean_df)
    
    processed_df.to_csv(output_path, index=False)
    print(f"[ETL] Processed dataset saved successfully to: {output_path}")
    return processed_df


# Update the bottom line in src/data_pipeline.py to:
if __name__ == "__main__":
    run_pipeline(raw_path="data/raw/cs-training.csv")