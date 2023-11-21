import os
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


path = "C:/Users/Big JC/PycharmProjects/Scraping/Export_from_Databases/scraping_scripts/"
filename = "scraped_files/origami_pdf_total_16122020.csv"
file = path + filename
remove_empty_lines(file)
# remove_raw_strings(file)
