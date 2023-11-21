import os
import pandas as pd
import numpy as np


def remove_empty_lines(filename):
    if not os.path.isfile(filename):
        print("{} does not exist ".format(filename))
        return
    with open(filename) as file_handle:
        lines = file_handle.readlines()

    with open(filename, 'w') as file_handle:
        lines = filter(lambda x: x.strip(), lines)
        file_handle.writelines(lines)


file = "GoogleScholar_files/Variational Autoencoder VAE Generation Arxiv.csv"
remove_empty_lines(file)
df = pd.read_csv(file, sep=',')  # encoding="ISO-8859-1"
df = df.iloc[:, 2]
df.index = np.arange(1, len(df) + 1)
print(df)


# Save Data Frame
file_name = 'GoogleScholar_url_list_2' + '.csv'  # Enter file name

df.to_csv(r'\Scraping\GoogleScholarSpecificScrape' + '\\' + file_name,
          index=None, header=True)
