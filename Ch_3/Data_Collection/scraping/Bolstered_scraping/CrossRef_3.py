from bs4 import BeautifulSoup
import requests
import pandas as pd
import time


path = "C:/Users/Big JC/PycharmProjects/Scraping/CrossRefExpansiveScrape/"
filename = "url_AllBiomaterial_1.csv"
file = path + filename
df = pd.read_csv(file)


def find_pdf_href():
    for row in df.itertuples():
        print("--- ORIGINAL LINK ---", row._1)
        base = str(row._1).split("/", 3)
        BASE_URL = (base[0]+"//"+base[2])
        time.sleep(0.5)

        try:
            page = requests.get(row._1).content
            soup = BeautifulSoup(page, 'html.parser')
            links = soup.find_all('a')
            links = [a for a in links if a.attrs.get('href') and 'pdf' in a.attrs.get('href')]
            print('-- pdf --')
            for idx, link in enumerate(links):
                print('{}) {}'.format(idx, link))
            # urls = ['{}{}'.format(BASE_URL, a.attrs.get('href')) for a in links if
            #         a.attrs.get('href') and 'suppl' in a.attrs.get('href')]
            # print('-- sup --')
            # for idx, url in enumerate(urls):
            #     print('{}) {}'.format(idx, url))

            # -------- Save Data frame ---------
            # link = pd.DataFrame({'original': [tag_link_list],
            #                      'full': [tag_full_list],
            #                      'pdf': [tag_pdf_list]})
            #
            # with open("link_logger2.csv", mode='a') as df_file:
            #     link.to_csv(df_file, header=None)

        except:
            print("Break")
            print(" ")


def return_redirect_link():
    for row in df.itertuples():
        r = requests.head(row._1, allow_redirects=True)
        redirect_url = r.url
        redirect = pd.DataFrame({'redirect': [redirect_url]})
        print(redirect)
        # Save Data Frame
        with open("Origami1_redirect.csv", mode='a') as df_file:
            redirect.to_csv(df_file, header=None)


return_redirect_link()
