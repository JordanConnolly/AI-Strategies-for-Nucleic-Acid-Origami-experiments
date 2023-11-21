import os
from tika import parser

# Save pdf as readable text
pdf = 'C:/Users/Big JC/PycharmProjects/Scraping/' \
       'Table_Retrieval/3_org_doi_suppl_10_1021_acssynbio_6b00271_suppl_file_sb6b00271_si_001_pdf.pdf'


def extract_text_from_pdf_recursively(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            path_to_pdf = os.path.join(root, file)
            [stem, ext] = os.path.splitext(path_to_pdf)
            if ext == '.pdf':
                print("Processing " + path_to_pdf)
                pdf_contents = parser.from_file(path_to_pdf)
                pdf_contents = pdf_contents['content']
                path_to_txt = stem + '.txt'
                with open(path_to_txt, 'w', encoding='utf-8') as txt_file:
                    print("Writing contents to " + path_to_txt)
                    txt_file.write(pdf_contents)


def read_text_from_pdf_recursively(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            path_to_pdf = os.path.join(root, file)
            [stem, ext] = os.path.splitext(path_to_pdf)
            if ext == '.pdf':
                print("Processing " + path_to_pdf)
                pdf_contents = parser.from_file(path_to_pdf)
                pdf_contents = pdf_contents['content']
                path_to_txt = stem + '.txt'
                pdf_contents.encode('utf-8')
                print(pdf_contents)


if __name__ == "__main__":
    extract_text_from_pdf_recursively(os.getcwd())
