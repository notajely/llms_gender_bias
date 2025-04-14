import pandas as pd

def load_stereotype_groups(bls_csv_path):
    """
    Load BLS gender ratio data and classify occupations into male-stereotyped, 
    female-stereotyped, and neutral categories based on the 'stereotype_label' column.
    
    Args:
        bls_csv_path (str): Path to the BLS gender ratio CSV file.
    
    Returns:
        tuple: Lists of male-stereotyped, female-stereotyped, and neutral occupations.
    """
    male_stereotype = []
    female_stereotype = []
    neutral = []
    
    df = pd.read_csv(bls_csv_path)
    if 'stereotype_label' not in df.columns:
        raise ValueError("The CSV file must contain a 'stereotype_label' column.")

    for _, row in df.iterrows():
        if row['stereotype_label'] == 'male-stereotyped':
            male_stereotype.append(row['bib'])
        elif row['stereotype_label'] == 'female-stereotyped':
            female_stereotype.append(row['bib'])
        elif row['stereotype_label'] == 'neutral':
            neutral.append(row['bib'])

    print(f"Loaded {len(male_stereotype)} male-stereotyped occupations.")
    print(f"Loaded {len(female_stereotype)} female-stereotyped occupations.")
    print(f"Loaded {len(neutral)} neutral occupations.")

    return male_stereotype, female_stereotype, neutral
