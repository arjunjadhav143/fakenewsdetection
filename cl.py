import pandas as pd

# Define the input and output file names
INPUT_FILE = 'news.csv'
OUTPUT_FILE = 'news_cleaned.csv'

print(f"Starting data cleaning process for '{INPUT_FILE}'...")

try:
    # Load the dataset
    df = pd.read_csv(INPUT_FILE)
    print("\nDataset loaded successfully. Original shape:", df.shape)
    print("Original columns:", df.columns.tolist())

    # --- 1. Remove the first unnamed column ---
    # This column is usually an old index from a previous save.
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
        print(f"\nRemoved 'Unnamed: 0' column.")
        print("Columns after removal:", df.columns.tolist())
    else:
        print("\nNo 'Unnamed: 0' column found to remove.")

    # --- 2. Handle missing values ---
    # We will remove any rows where 'title', 'text', or 'label' is empty.
    print("\nChecking for missing values...")
    print(df.isnull().sum())
    
    original_rows = len(df)
    df.dropna(inplace=True)
    rows_after_dropna = len(df)
    print(f"\nRemoved {original_rows - rows_after_dropna} rows with missing values.")

    # --- 3. Save the cleaned data ---
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nCleaning complete! The organized data has been saved to '{OUTPUT_FILE}'.")
    print("Final shape of the cleaned data:", df.shape)

except FileNotFoundError:
    print(f"\nERROR: The file '{INPUT_FILE}' was not found. Please make sure it's in the same folder as this script.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
