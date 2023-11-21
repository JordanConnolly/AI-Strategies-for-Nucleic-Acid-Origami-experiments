import requests
from bs4 import BeautifulSoup
import re
import time
import random
import pandas as pd

query = input("Type in the console your google scholar search query")
# Input spaces as +

page_number = 0
tag_number = 0

url = ("https://scholar.google.co.uk/scholar?start=" + str(page_number) + "&q=" + query + "&hl=en&as_sdt=0,5")

for urls in url:
    tag_number += 10
    url = ("https://scholar.google.co.uk/scholar?start=" + str(page_number) + "&q=" + query +
           "&hl=en&as_sdt=0,5")
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    print(url)

    # for tags in soup.find_all("div", class_="gs_a"):
    #     Journal = tags.find('a')
    #     print(tags)
    #     print(Journal)
    #     Journals = pd.DataFrame({'Journal'})
    #     time.sleep(random.uniform(0.5, 1.0))

    for tag in soup.find_all(re.compile("^h3")):
        title = tag.find('a')

        try:
            title_text = title.get_text("").encode("utf-8")
        except AttributeError:
            title_text = "Title Error: -"

        try:
            hyperlink_text = title.get('href')
        except AttributeError:
            hyperlink_text = "Hyperlink Error: -"

        df = pd.DataFrame({'title': [title_text],
                           'hyperlink': [hyperlink_text]})
        print(df)

        http = str(df['hyperlink']).find('http')
        if http > 5:
            raise SystemExit
        if title is None:
            raise SystemExit

        print("- - - - -")

        with open(query + ".csv", mode='a') as df_file:
            df.to_csv(df_file, header=None)

        if tag_number % 10 == 0:
            page_number = tag_number
            # time.sleep(random.uniform(1.0, 1.5))

    time.sleep(random.uniform(1.0, 5.0))
    print("- - - - - Running - - - - -")
