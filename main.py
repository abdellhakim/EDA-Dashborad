import os
import pandas as pd
from eda_modules.load_data import load_and_preprocess
from eda_modules.summary import generate_summary
from eda_modules.correlation import generate_correlation
from eda_modules.forecast import forecast_sales
from eda_modules.patterns import detect_outliers

DATA_FOLDER = "data"

def list_files():
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    if not files:
        print("⚠️ No CSV files found in 'data/' folder.")
        return None
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")
    return files

def main():
    print("📂 Available CSV files:")
    files = list_files()
    if not files:
        return

    file_index = int(input("Select a file by number: ")) - 1
    filename = files[file_index]
    file_path = os.path.join(DATA_FOLDER, filename)

    df = load_and_preprocess(file_path)

    print("\n📊 Available EDA Tools:")
    tools = ["summary", "correlation", "forecast", "outliers"]
    for i, tool in enumerate(tools):
        print(f"{i+1}. {tool}")

    tool_index = int(input("Select an analysis to perform: ")) - 1
    choice = tools[tool_index]

    print("\n🔍 Result:\n")
    if choice == "summary":
        print(generate_summary(df))
    elif choice == "correlation":
        print(generate_correlation(df))
    elif choice == "forecast":
        print(forecast_sales(df))
    elif choice == "outliers":
        print(detect_outliers(df))
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
