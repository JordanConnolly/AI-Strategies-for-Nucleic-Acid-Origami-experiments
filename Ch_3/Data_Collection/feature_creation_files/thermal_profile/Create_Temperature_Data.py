import pandas as pd
import numpy as np
import os
import csv
from Thermal_Creator import incubated, stepwise, linear

path = "creation_of_thermal_profile_times/Temperature_Curves/"
Incubated = pd.read_excel('Curve_Types_Split.xlsx', sheet_name='Type_1_Curves')
Linear = pd.read_excel('Curve_Types_Split.xlsx', sheet_name='Type_2_Curves')
Stepwise = pd.read_excel('Curve_Types_Split.xlsx', sheet_name='Type_3_Curves')

label_list = []


def create_label(spreadsheet):
    """Reads the spreadsheet and creates ID based label from columns"""
    # Use these as labels for the CSV files
    print("Column headings:", spreadsheet.columns)
    for index, row in spreadsheet.iterrows():
        sort_id = row['CURVE_ID']
        structure_id = row['YIELD_NUMBER']
        label_w_space = "label_" + str(sort_id) + "_" + str(structure_id)
        label_wo_space = str(label_w_space).strip()
        label_list.append(label_wo_space)


def create_incubated():
    """Creates the incubated curves"""
    results = []
    count = 0
    for row in Incubated.itertuples():
        peak = getattr(row, "PEAK")
        rate = getattr(row, "RATE")
        file_name = label_list[count]
        print(file_name)
        count += 1
        column = str(count)
        data = incubated(peak, rate)
        print(column)
        # df1 = df1.append({'key_%s' % count: j}, ignore_index=True)
        df1 = pd.DataFrame()
        df1['key'] = data
        print(df1.head())
        df1.to_csv(os.path.join(path, file_name + ".csv"))


def create_linear():
    """Creates the linear curves"""
    results = []
    count = 0
    for row in Linear.itertuples():
        peak = getattr(row, "PEAK")
        base = getattr(row, "BASE")
        rate = getattr(row, "RATE")
        file_name = label_list[count]
        print(file_name)
        count += 1
        column = str(count)
        data = linear(peak, base, rate)
        print(column)
        # df1 = df1.append({'key_%s' % count: j}, ignore_index=True)
        df2 = pd.DataFrame()
        df2['key'] = data
        print(df2.head())
        df2.to_csv(os.path.join(path, file_name + ".csv"))


def create_stepwise():
    """Creates the stepwise curves"""
    count = 0
    for row in Stepwise.itertuples():
        peak = getattr(row, "PEAK")
        base = getattr(row, "BASE")
        rate = getattr(row, "RATE")
        modulo = getattr(row, "MODULO")
        file_name = label_list[count]
        print(file_name)
        count += 1
        column = str(count)
        data = stepwise(peak, base, rate, modulo)
        print(column)
        # df1 = df1.append({'key_%s' % count: j}, ignore_index=True)
        df3 = pd.DataFrame()
        df3['key'] = data
        print(df3.head())
        df3.to_csv(os.path.join(path, file_name + ".csv"))


# create_label(spreadsheet=Incubated)
# create_incubated()

# create_label(spreadsheet=Linear)
# create_linear()

create_label(spreadsheet=Stepwise)
create_stepwise()
