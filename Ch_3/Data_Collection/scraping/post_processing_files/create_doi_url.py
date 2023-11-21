import os
import pandas as pd


def create_url(file_used):
    if not os.path.isfile(file_used):
        print("{} does not exist ".format(file_used))
        return
    with open(file_used) as file_handle:
        lines = file_handle.readlines()
    with open("url_" + file_used, 'w') as file_handle:
        for line in lines:
            file_handle.write("https://doi.org/" + line)


path = "C:/Users/Big JC/PycharmProjects/Scraping/Export_from_Databases/database_files/"
filename = "../database_files/master_DNA_Origami_papers_16122020.csv"
file = path + filename


def create_url_pandas(file_used):
    doi_containing_file = pd.read_csv(file_used)
    doi_containing_file['DOI_URL'] = "https://doi.org/" + doi_containing_file['DOI'].astype(str)
    doi_containing_file.to_csv(filename, index=False)
    return


create_url_pandas(file)
