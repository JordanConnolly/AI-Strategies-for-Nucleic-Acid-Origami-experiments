import os.path
import glob
import re
import os

# Create a load of text files for me to fill with Methods
path = "C:/Users/Big JC/PycharmProjects/Scraping/Export_from_Databases/database_files/Method_Files/"
file_name = "paper_method_"

# # Use a pre-set range (naive)
# for i in range(100, 1903):
#     complete_name = os.path.join(path + file_name + str(i) + ".txt")
#     file1 = open(complete_name, "w")
#     file1.close()

# Use the PDFs that exist with their index as a file name
numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# Directories containing files
cwd = os.getcwd()
# Remove last directory
split_cwd = cwd.split("\\")[-1]
new_wd = cwd[:-len(split_cwd)]
# # directory of interest
# interest = "database_files/PDF_Files/"
# path_to_file = new_wd + interest
# # iterate to create a list of files of interest
# all_files = sorted(glob.glob(path_to_file + "pdf_main_" + "*" + ".pdf"), key=numerical_sort)
# print("number of pdf files stored:", len(all_files))
#
# for file in all_files:
#     split_filename = file.split("\\")[-1]
#     file_type = split_filename.split("_")[-1]
#     file_index = file_type.split(".")[0]
#     complete_name = os.path.join(path + file_name + str(file_index) + ".txt")
#     file1 = open(complete_name, "w")
#     file1.close()

# append files with a structure
# directory storing text files
store = "database_files/Method_Files/"
path_to_stored_files = new_wd + store
# iterate to append all files
txt_files = sorted(glob.glob(path_to_stored_files + "paper_method_" + "*" + ".txt"), key=numerical_sort)

# # over-write files - do not uncomment and run unless you want to wipe out content
# for txt_file in txt_files:
#     # open file and append
#     with open(txt_file, 'w') as the_file:
#         the_file.write('Main:\n'
#                        '/\n'
#                        'Suppl:\n'
#                        '/\n')
