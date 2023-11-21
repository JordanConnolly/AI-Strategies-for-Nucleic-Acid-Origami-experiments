import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.request import Request, urlopen
import re
import time

link_list = pd.read_csv("DOI_Logger_Broken_3.csv")


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


text_regex = r"(?i)DOI:\s*[1][0][.]\d\d\d\d\/[^\s]+"
link_regex = r"(?i)(https\:\/\/doi[^\s]+|http\:\/\/doi[^\s]+)"

for row in link_list.itertuples():
    try:
        req = Request(row._1, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        text = text_from_html(html)
        matches = re.findall(text_regex, text)
        link_matches = re.findall(link_regex, text)

        DOI_list = []
        DOI_link_list = []
        row_list = []

        DOI_list.append(matches)
        DOI_link_list.append(link_matches)
        row_list.append(row)

        print("Original: ", row._1)
        print("text: ", matches)
        print("links: ", link_matches)
        # time.sleep(3)

        row = row.replace('\n', '')
        link = pd.DataFrame({'original': [row], 'text': [DOI_list], 'link': [DOI_link_list]})

        with open("DOI_List_Broken_Fix.csv", mode='a') as df_file:
            link.to_csv(df_file, header=False, index=None, line_terminator='\n')
    except:
        print("Error:")
        print(row)


# # https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text