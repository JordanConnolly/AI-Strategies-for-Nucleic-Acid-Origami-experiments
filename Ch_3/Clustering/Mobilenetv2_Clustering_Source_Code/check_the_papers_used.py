import pandas as pd


df = pd.read_csv('image_paper_number.csv')

# Count the number of unique rows
num_unique = df.nunique()

# Print the result
print('Number of unique rows:', num_unique)
