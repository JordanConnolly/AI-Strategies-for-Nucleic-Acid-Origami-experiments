import pandas as pd
import re
import glob
import numpy as np


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def gc_percentage_creator(df):
    """Print GC counts"""
    gc_count_list = []
    for staple in df:
        print(staple)
        string = str(staple)
        gc = (string.count("G") + string.count("C"))
        at = (string.count("A") + string.count("T"))
        if gc != 0 and at != 0:
            gc_percentage_calc = gc / (gc+at)
            gc_percentage = round(gc_percentage_calc, 3)
            gc_count_list.append(gc_percentage)
        else:
            gc_count_list.append(0)
    gc_data = pd.DataFrame(gc_count_list, columns=["gc"])
    print(gc_data)
    return gc_data


"""Read in all of the staple data for each experiment;
create G/C%
create Metrics (basic stats)
collect number of staples per structure
No. modified strands (report but don't collect DNA/RNA No.)
Material Type (over-all, DNA, RNA, Hybrid)
Close and store data columns for each yield number"""

# create a large data frame of staples
path = "/final_stage_dataset/" \
       "final_stage_staples_for_post_100_literature_papers/staple_sequences_with_og_collection/"

# loop over text files and create staples data-frame
staple_file_naming = "staple_" + "*.txt"
all_staple_files = sorted(glob.glob(path + staple_file_naming), key=numerical_sort)

staple_files = []
for staple_file in all_staple_files:
    staple_file_isolated = staple_file.split("/")[-1]
    staple_file_number = staple_file_isolated.split(".")[0].split("_")[-1]
    if staple_file_number != "unmodified":
        if int(staple_file_number) < 10000:
            staple_files.append(staple_file)

staples_df = pd.DataFrame()
for i in staple_files:
    # open file to file staple
    completeName = i
    # print(completeName)
    f = open(completeName, "r")
    lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    f.close()

    # get txt file name and use as paper number and experiment number columns
    filename = i.split("\\")[-1:]
    # print(filename)
    filename = filename[0]
    paper = filename.split("_")[3]
    experiment = filename.split("_")[5]
    experiment = experiment.split(".")[0]
    temp_staples_df = pd.DataFrame()
    temp_staples_df['STAPLE'] = lines
    temp_staples_df['Experiment Number'] = experiment
    temp_staples_df['Paper Number'] = paper
    if len(lines) > 0:
        temp_staples_df['gc'] = gc_percentage_creator(temp_staples_df['STAPLE'])
        temp_staples_df['STAPLE_NT_LENGTH'] = temp_staples_df['STAPLE'].str.len()
    else:
        temp_staples_df['gc'] = 0
        temp_staples_df['STAPLE_NT_LENGTH'] = 0

    staples_df = pd.concat([staples_df, temp_staples_df])


print(staples_df["Experiment Number"])
# staples_df.to_csv("all_staples_all_literature.csv")

structures_staple = staples_df.groupby(('Experiment Number'), sort=False)['STAPLE'].\
    apply(lambda group_series: group_series.tolist())
# staples_df_grouped_count = staples_df.groupby('YIELD_NUMBER')['STAPLE'].apply(list)
structures_df = pd.DataFrame(structures_staple)

# print(staples_df['YIELD_NUMBER'])
structures_length = staples_df.groupby(('Experiment Number'), sort=False)['STAPLE_NT_LENGTH'].\
    describe().add_prefix("staple_").reset_index()
print(structures_length.dtypes)
cols = structures_length.columns[structures_length.dtypes.eq('float64')]
structures_length_round = structures_length.round(decimals=0)
print(structures_length_round)
regex = re.compile(r'(-?)(\d)')


def rna_or_dna(df):
    """print material type"""
    rna_or_dna_list = []
    for staple_set in df:
        for staple in staple_set:
            string = str(staple)
            if 'T' in string:
                rna_or_dna_list.append(0)
            elif 'U' in string:
                rna_or_dna_list.append(1)


def modified_check(df):
    """Check if there are modified strands"""
    modified_list = []
    for staple_set in df:
        for staple in staple_set:
            string = str(staple)
            if regex.search(string) is None:
                modified_list.append(0)
            else:
                modified_list.append(1)
    print(modified_list)


gc_description = staples_df.groupby(('Experiment Number'), sort=False)['gc'].describe().add_prefix("gc_").reset_index()
gc_description_round = gc_description.round(decimals=3)
final_description = pd.concat([structures_length_round, gc_description_round], axis=1)

# # fit descriptions in correct order
# data_frame_of_ml_instances = pd.read_excel("all_instances_literature_dataset_in_progress.xls",
#                                            usecols="C")
# print(data_frame_of_ml_instances)

print(final_description)
descriptive_file = "new_bolstered_descriptive_staple_file_all_literature.csv"
final_description.to_csv(descriptive_file)
