import requests
import PyPDF4
import pandas as pd

pdf_writer = PyPDF4.PdfFileWriter()
directory_path_pdf = "C:/Users/Big JC/PycharmProjects/Scraping/Export_from_Databases/database_files/" \
                     "master_DNA_Origami_papers_16122020.csv"


def download_pdf_files(file_path):
    if __name__ == '__main__':
        try:
            count = 1
            pdf_file = pd.read_csv(file_path)
            print(pdf_file.columns)
            pdf_file_url = pdf_file['pdf_link_2']
            paper_url = pdf_file['Redirect_links']
            # print(paper_url)
            landing_url_list = []
            # for pdf_url in pdf_file_url:
            # print(pdf_url)
            # with open(str(pdf_count) + "_" + pdf_name + '.pdf', 'wb') as f:
            #     response = requests.get(pdf_urls)
            #     f.write(response.content)
            #     print("Original File: ", pdf_name, count)
            #     count += 1
        except:
            print("failed")


def download_origin_url(file_path):
    if __name__ == '__main__':
        count = 1
        pdf_file = pd.read_csv(file_path)
        pdf_file_url = pdf_file['pdf_link_2']
        paper_url = pdf_file['Redirect_links']
        landing_url_list = []
        for url in paper_url:
            base_url = url.split("/")[:3]
            landing_url = base_url[0] + "//" + base_url[2]
            landing_url_list.append(landing_url)
        landing_url_df = pd.DataFrame(landing_url_list)
        landing_url_df.to_csv("landing_url_list.csv")


def combine_origin_url(file_path, origin_path):
    pdf_file = pd.read_csv(file_path)
    pdf_file_url = pdf_file['pdf_link_2']
    origin_file = pd.read_csv(origin_path)
    original_url = origin_file['0']
    combined_url_list = []
    for url in pdf_file_url:
        for url_origin in original_url:
            print(url)


# download_origin_url(directory_path_pdf)
original_path = "C:/Users/Big JC/PycharmProjects/Scraping/Export_from_Databases/scraping_scripts/landing_url_list.csv"
combine_origin_url(directory_path_pdf, original_path)

