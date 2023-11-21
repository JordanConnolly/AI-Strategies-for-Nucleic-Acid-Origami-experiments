import re
import os

# Tags the Text to allow filtering based on keywords
regex = 'https://arxiv.org/', 'biorxiv'

combined_regex = re.compile('|'.join('(?:{0})'.format(x) for x in regex))


def read_write_match_file(filename):
    if not os.path.isfile(filename):
        print("{} does not exist ".format(filename))
        return

    with open(filename, 'r') as file:
        number = 0
        matched = 0

        with open("GoogleScholar_files/VAE_labelled.csv", 'a') as write_file:

            for files in file:
                matches = re.findall(combined_regex, files)  # Regex
                number += 1
                # print("Count: ", number)
                if matches:
                    write_file.writelines(files)
                    matched += 1
                    print(number, matches, matched, files)
            write_file.close()


read_write_match_file("GoogleScholar_files/GoogleScholar_url_list_2.csv")
