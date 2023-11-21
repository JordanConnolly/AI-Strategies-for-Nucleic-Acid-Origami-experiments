import os
import re
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

# Tags the Text to allow filtering based on keywords
regex = 'p8064'

combined_regex = re.compile('|'.join('(?:{0})'.format(x) for x in regex))

print("pdf")


def analyse_text_from_pdf_recursively(directory):
    for root, dirs, files in os.walk(directory):
        count = 0
        file_names = []
        for file in files:
            path_to_pdf = os.path.join(root, file)
            [stem, ext] = os.path.splitext(path_to_pdf)
            if ext == '.txt':
                file_names.append(stem)
                txt_list = []
                count += 1
                with open(file, encoding='utf-8') as f:
                    txt_list.append(f.read())
                    matches = re.findall(regex, str(txt_list))
                    if matches:
                        match_name = (str(matches))
                        counts = dict()
                        words = match_name.split()
                        for word in words:
                            if word in counts:
                                counts[word] += 1
                            else:
                                counts[word] = 1

                        total_counts = sum(counts.values())
                        print(match_name, total_counts, file)
                        df = pd.DataFrame({'file_name': [str(file)[0:3]], 'match_count': [str(total_counts)]})
                        with open(regex + "_pdf_" + "1" + ".csv", mode='a') as df_file:
                            df.to_csv(df_file, header=False, index=False)


def plot_bar_chart(df_file):
    with open(df_file) as file_handle:
        lines = file_handle.readlines()
    with open(df_file, 'w') as file_handle:
        lines = filter(lambda x: x.strip(), lines)
        file_handle.writelines(lines)

    data_set = pd.read_csv(df_file, names=['file_name', 'match_count'])
    data_set = data_set.dropna()
    print(data_set.head(5))
    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure()
        ax = sns.barplot(x="match_count", y="file_name", data=data_set, orient='h', palette="OrRd_d")
        ax.set_xlabel('matching word count')
        ax.set_title('Number of matching words across files')
        ax.grid(True)
        plt.show()


if __name__ == "__main__":
    analyse_text_from_pdf_recursively("C:/Users/Big JC/PycharmProjects/Scraping/text_retrieval")
    # plot_bar_chart("well-formed1.csv")
    # analyse_text_from_pdf_recursively(os.getcwd())
