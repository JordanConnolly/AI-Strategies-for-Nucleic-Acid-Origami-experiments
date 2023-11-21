import random
import time
import pandas as pd
from crossref.restful import Works

# .filter(from_online_pub_date='2017') - Can be used for filtering by date
# publisher_name='ACS Synthetic Biology'

works = Works()

#  Search Terms / queries of interest
# x = ("Nanostructures", "DNA", "RNA", "Multi-layer", "Nucleic", "Objects", "Nanoscale", "Biomaterial")

x = ("Origami", "origami")
#  This finds the number of Titles related to X queries and returns it
for titles in x:
    w1 = works.query(container_title=titles).filter(from_online_pub_date='2006').count()
    print(titles + ": " + str(w1))

w1 = works.query(container_title=x).filter(from_online_pub_date='2006').count()
print("all together: " + str(w1))


#  This finds the DOI related to X queries and returns it
# for titles in x:
#     w2 = works.query(container_title=titles).filter(from_online_pub_date='2006').sort('published').order('asc')
#     for item in w2:
#         try:
#             # DOI_text = DOI_text[0].encode("utf-8")
#             DOI_text = DOI = item['DOI']
#             URL = item['URL']
#             # title_text = title = item['title']
#             # publisher_text = publisher = item['publisher']
#             # print(publisher_text)
#             # print(title_text)
#             print(DOI_text)
#             # time.sleep(random.uniform(0.1, 0.5))
#         except KeyError:
#             DOI_text = "DOI Error: -"
#             URL = "URL Error: -"
#             # title_text = "Title Error: -"
#
#         df = pd.DataFrame({'DOI': [DOI_text], 'URL': [URL]})
#         with open(titles + "1" + ".csv", mode='a') as df_file:
#             df.to_csv(df_file, header=None, index=None)


# This finds the Details related to query w3 and returns it
for titles in x:
    w3 = works.query(container_title=titles).filter(from_online_pub_date='2006').sort('published').order('asc')
    counter = 0
    print(titles)
    for item in w3:
        counter += 1
        try:
            title_text = title = item['title']
            title_text = str(title_text)
            title_text = title_text.encode("utf-8")
        except KeyError:
            title_text = "Title Error: -"

        # year = item['published']
        # date = item['deposited']

        try:
            issued_text = issued = item['journal-issue']
        except KeyError:
            issued_text = "Issued Error: -"

        try:
            reference_used_text = item['reference-count']
        except KeyError:
            reference_used_text = "Ref Error: -"

        try:
            publisher_text = publisher = item['publisher']
        except KeyError:
            publisher_text = "Publisher Error: -"

        try:
            DOI_text = DOI = item['DOI']
        except KeyError:
            DOI_text = "DOI Error: -"

        try:
            ref_count_text = ref_count = item['is-referenced-by-count']
        except KeyError:
            ref_count_text = "Is Ref Error: -"

        try:
            container_text = container = item['short-container-title']
            container_text = str(container_text)
            container_text = container_text.encode("utf-8")
        except KeyError:
            container_text = "Cont Title Error: -"

        # full_info = item
        # print(full_info)
        # time.sleep(random.uniform(0.01, 1.0))

        print(counter)
        df = pd.DataFrame({'title': [title_text], 'DOI': [DOI_text], 'publisher': [publisher_text],
                           'container': [container_text],
                           'reference_used': [reference_used_text], 'ref_count': [ref_count_text],
                           'issued': [issued_text]})

        with open(titles + "-CrossRef-Titles.csv", mode='a') as df_file:
            df.to_csv(df_file, header=None, index=None)
