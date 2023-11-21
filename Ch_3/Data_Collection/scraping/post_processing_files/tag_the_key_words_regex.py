import re
import os

# Tags the Text to allow filtering based on keywords
# regex = 'DNA', 'RNA', 'Origami',\
#         '3D ', '2D ', 'Dimensional', 'dimensional', ' Two', ' Three', ' two', ' three', 'origami', 'Two', 'Three',\
#         '3D-', '2D-', 'AFM', 'Atomic Force Microscopy', 'atomic force microscopy', 'ML', 'Machine Learning',\
#         'SPM', 'Scanning Probe Microscopy', 'scanning probe microscopy', 'Nanostructures', \
#         'Multi-layer', 'Nucleic', 'Objects', 'Nanoscale', 'Biomaterial'

# Tag the text for lab informing information
regex = '3D ', '2D ', 'Dimensional', 'dimensional', ' Two', ' Three', ' two', ' three', 'origami', 'Two', 'Three',\
        '3D-', '2D-', 'Magnesium', 'Mg', 'EDTA', 'PBS', 'Folding', 'Buffer', 'folding', 'buffer', 'edta', 'thermal',\
        'Thermal'

combined_regex = re.compile('|'.join('(?:{0})'.format(x) for x in regex))


def read_write_match_file(filename):
    if not os.path.isfile(filename):
        print("{} does not exist ".format(filename))
        return
    with open(filename, 'r') as file:
        number = 0
        matched = 0
        # file_name = filename.split("/")
        # file_split = file_name[1].split(".")
        # file_string = str(file_split[0])
        # print(file_string)
        with open("Store_files_scraped_3/Origami1_CrossRef_labelled.csv", 'a') as write_file:

            for files in file:
                matches = re.findall(combined_regex, files)  # Combined Regex (multiple)
                # matches = re.findall(regex, files)  # Regex for single words
                number += 1
                # print("Count: ", number)
                if matches:
                    match_name = (str(matches))
                    match_name = match_name.replace(',', '.')
                    match_name = (match_name + " ,")
                    write_file.writelines(match_name)
                    write_file.writelines(files)
                    matched += 1
                    print(number, matches, matched, files)
            write_file.close()


read_write_match_file("Store_files_scraped_3/DNA-CrossRef-Titles.csv")
