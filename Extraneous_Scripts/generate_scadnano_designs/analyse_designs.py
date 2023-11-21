import pandas as pd
import numpy as np
import glob
import re
import scadnano as sc
from useful_code_snippets import *


# design storage location
storage_path = "output_designs/"
design_storage = storage_path + "design/"
sequence_storage = storage_path + "seqs/"

sequence_file_names = sorted(glob.glob(sequence_storage + "*.xls"), key=numerical_sort)
print("sequence files in folder:", len(sequence_file_names))

design_file_names = sorted(glob.glob(design_storage + "*.sc"), key=numerical_sort)
print("design files in folder:", len(design_file_names))

for designs in design_file_names:
    # designs = designs.split("\\")[1]
    # print(designs)
    # open designs into scadnano
    design = sc.Design().from_scadnano_file(designs)
    # Iterate over staple strands and add to list
    strand_len_list = []
    for strand in design.strands:
        if not strand.is_scaffold:
            individual_staple_len = len(strand.dna_sequence)
            strand_len_list.append(individual_staple_len)
    print(np.average(strand_len_list))
