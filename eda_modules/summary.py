def generate_summary(df):
    summary = []

    # Dataset shape
    summary.append(f"🧾 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Column names and data types
    summary.append("📌 Column info:")
    for col in df.columns:
        summary.append(f"  - {col}: {df[col].dtype} (non-null: {df[col].notnull().sum()})")
    summary.append("")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        summary.append("⚠️ Missing values:")
        for col, val in missing.items():
            summary.append(f"  - {col}: {val} missing")
        summary.append("")
    
    # Numerical summary
    summary.append("📊 Numerical summary:\n")
    summary.append(df.describe().to_string())
    summary.append("")

    # Categorical summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        summary.append("🔠 Categorical columns summary:")
        for col in cat_cols:
            summary.append(f"  - {col}: {df[col].nunique()} unique values")
            top_vals = df[col].value_counts().head(3)
            summary.append(f"    Top 3 values: {', '.join([f'{k} ({v})' for k,v in top_vals.items()])}")
        summary.append("")

    return '\n'.join(summary)
