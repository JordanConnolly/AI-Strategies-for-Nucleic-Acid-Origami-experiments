from bs4 import BeautifulSoup
import requests
import pandas as pd
import time


path = "C:/Users/Big JC/PycharmProjects/Scraping/Export_from_Databases/database_files/"
filename = "../database_files/master_DNA_Origami_papers_16122020.csv"
file = path + filename
df = pd.read_csv(file)
# df = df['DOI_URL']  # the DOI url for use in return_redirect_link
redirect_url_df = df['Redirect_links']  # if return_redirect_link


def find_pdf_href():
    total_list = []
    counter = 0
    for row in redirect_url_df:
        counter += 1
        print("--- ORIGINAL LINK ---", row)
        time.sleep(6)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }
        page = requests.get(row, headers=headers).content
        soup = BeautifulSoup(page, 'html.parser')
        links = soup.find_all('a', href=True)
        # links = [a for a in links if a.attrs.get('href') and 'pdf' in a.attrs.get('href')]
        print('-- pdf --')
        pdf_list = []
        for a in soup.find_all('a', href=True):
            if 'pdf' in a.attrs.get('href'):
                # print(a['href'])
                pdf_list.append(a['href'])
        total_list.append(pdf_list)

        inner_counter = 0
        for pdfs in pdf_list:
            inner_counter += 1
            print(f"Fetching pdf: {counter}...")
            with open(f"{counter}_{inner_counter}.pdf", "wb") as f:
                f.write(requests.get(pdfs).content)

    # -------- Save Data frame ---------
    link = pd.DataFrame(total_list)
    with open("scraped_files/origami_pdf_total_21122020.csv", mode='a') as df_file:
        link.to_csv(df_file, header=None)


def return_redirect_link():
    for row in df:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }
        r = requests.head(row, headers=headers, allow_redirects=True)
        redirect_url = r.url
        redirect = pd.DataFrame({'redirect': [redirect_url]})
        print(redirect)
        time.sleep(1)
        # Save Data Frame
        with open("scraped_files/Origami_redirect_headers.csv", mode='a', newline='') as df_file:
            redirect.to_csv(df_file, header=None)


# return_redirect_link()
find_pdf_href()