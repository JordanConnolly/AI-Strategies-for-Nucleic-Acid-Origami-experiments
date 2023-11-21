import random
import time
import pandas as pd
from crossref.restful import Works
from unidecode import unidecode
import sys
import json
import requests


url = 'http://httpbin.org/status/200'
r = requests.get(url)

if 'json' in r.headers.get('Content-Type'):
    js = r.json()
else:
    print('Response content is not in JSON format.')
    js = 'spam'


def remove_non_ascii(text):
    return unidecode(str(text))


# .filter(from_online_pub_date='2017') - Can be used for filtering by date
# publisher_name='ACS Synthetic Biology'

works = Works()

#  Search Terms / queries of interest
x = ("Origami", "origami")

#  This finds the number of Titles related to X queries and returns it
for titles in x:
    print("all works")
    w1 = works.query(title=titles).filter(from_online_pub_date='2006').count()
    print(w1)
    print("specific publisher counts")
    w2 = works.query(title=titles, publisher_name='ACS').filter(from_online_pub_date='2006').count()
    print(w2)

# for titles in x:
#     w3 = works.query(container_title=titles).filter(from_online_pub_date='2006')
#     count = 0
#     for item in w3:
#         count += 1
#         print(count)
#
#         doi_text = item['DOI']
#         doi_text = str(doi_text)
#         # doi_text = doi_text.encode("utf-8")
#
#         # title_text = item['title']
#         # publisher_text = item['publisher']
#         # print(publisher_text)
#
#         # remove_non_ascii(title_text)
#
#         # title_text = str(title_text)
#         # title_text = title_text.encode("utf-8")
#
#         # df = pd.DataFrame({'doi': [doi_text], 'title': [title_text], 'publisher': [publisher_text]})
#         df = pd.DataFrame({'doi': [doi_text]})
#         with open("All" + titles + "_1" + ".csv", mode='a') as df_file:
#             df.to_csv(df_file, header=None, index=None)
