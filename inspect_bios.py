import pickle

# Load the pickle file
with open('data/raw/bios_train.pkl', 'rb') as f:
    data = pickle.load(f)

# Print basic information
print(f"Data type: {type(data)}")
print(f"Data length: {len(data)}")

# If it's a DataFrame, print column information
if hasattr(data, 'columns'):
    print(f"Columns: {data.columns.tolist()}")
    print("\nSample data (first 5 rows):")
    print(data.head())

    # Check if 'occupation' column exists
    if 'occupation' in data.columns:
        print("\nUnique occupations:")
        print(data['occupation'].value_counts().head(10))
else:
    print("Not a DataFrame")
