import pandas as pd
import numpy as np
import re
import glob
from useful_code_snippets import *

# create paths to the sheets
database_sheet1_path = "C:/Users/Big JC/PycharmProjects/Scraping/Export_from_Databases/database_files/" \
                       "master_detail_extraction_09052021.xlsx"
print(database_sheet1_path)
# read as data frame
database_sheet1_df = pd.read_excel(database_sheet1_path)
print(database_sheet1_df.columns)

