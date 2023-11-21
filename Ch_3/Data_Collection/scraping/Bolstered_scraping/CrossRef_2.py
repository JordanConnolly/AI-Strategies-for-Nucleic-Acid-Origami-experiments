import os
import pandas as pd
import numpy as np
import codecs


def remove_empty_lines(filename):
    if not os.path.isfile(filename):
        print("{} does not exist ".format(filename))
        return
    with open(filename) as file_handle:
        lines = file_handle.readlines()
    with open(filename, 'w') as file_handle:
        lines = filter(lambda x: x.strip(), lines)
        file_handle.writelines(lines)


def remove_raw_strings(filename):
    if not os.path.isfile(filename):
        return
    with open(filename) as file_handle:
        lines = file_handle.readlines()
        for line in lines:
            new_line = codecs.decode(line)
            print(new_line)


path = "C:/Users/Big JC/PycharmProjects/Scraping/CrossRefExpansiveScrape/"
filename = "Store_files_scraped_3/Origami-CrossRef-Titles.csv"
file = path + filename
remove_empty_lines(file)
# remove_raw_strings(file)

df = pd.read_csv(file)  # encoding="ISO-8859-1"
# df = df['doi']
df.index = np.arange(1, len(df) + 1)
print(df)


# def create_url(filename):
#     if not os.path.isfile(filename):
#         print("{} does not exist ".format(filename))
#         return
#     with open(filename) as file_handle:
#         lines = file_handle.readlines()
#     with open("url_" + filename, 'w') as file_handle:
#         for line in lines:
#             file_handle.write("https://doi.org/" + line)
#
#
# create_url(filename)
# url_file = path + "url_" + filename
# url_df = pd.read_csv(url_file)
# url_df.index = np.arange(1, len(df) + 1)
# print(url_df)
