import pickle
import pandas as pd
import csv

# Load the pickle file
with open('data/raw/bios_train.pkl', 'rb') as f:
    data = pickle.load(f)

# Load the occupation list
occupation_list = []
with open('data/bias_in_bia_occupation_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        occupation_list.append((int(row[0]), row[1]))

# Create a mapping from profession code to occupation name
profession_to_occupation = {code: name for code, name in occupation_list}

# Print the mapping
print("Profession code to occupation mapping:")
for code, name in sorted(profession_to_occupation.items()):
    print(f"{code}: {name}")

# Count occurrences of each profession in the dataset
profession_counts = data['profession'].value_counts().sort_index()
print("\nProfession counts in the dataset:")
for prof_code, count in profession_counts.items():
    occupation = profession_to_occupation.get(prof_code, "Unknown")
    print(f"Code {prof_code} ({occupation}): {count} samples")
