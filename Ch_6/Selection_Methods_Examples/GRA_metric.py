# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
# source GraModel: https://www.programmersought.com/article/49226244393/
# source: https://github.com/rsarai/grey-relational-analysis/blob/master/Gait%20and%20Grey%20Methods.ipynb


class GraModel():
    '''Gray correlation analysis model'''

    def __init__(self, inputData, p=0.5, standard=True):
        '''
                 Initialization parameters
                 inputData: input matrix, the vertical axis is the attribute name, the first column is the parent sequence
                 p: Resolution coefficient, range 0~1, generally 0.5, the smaller, the greater the difference between correlation coefficients, and the stronger the distinguish
                 standard: whether standardization is required
        '''
        self.inputData = np.array(inputData)
        self.p = p
        self.standard = standard
        # standardization
        self.standarOpt()
        # Modeling
        self.buildModel()

    def standarOpt(self):
        '''Standardized input data'''
        if not self.standard:
            return None
        self.scaler = StandardScaler().fit(self.inputData)
        self.inputData = self.scaler.transform(self.inputData)

    def buildModel(self):
        # The first column is the mother column, find the absolute difference with other columns
        momCol = self.inputData[:, 0]
        sonCol = self.inputData[:, 0:]
        for col in range(sonCol.shape[1]):
            sonCol[:, col] = abs(sonCol[:, col] - momCol)
        # Find the minimum difference and maximum difference between two levels
        minMin = sonCol.min()
        maxMax = sonCol.max()
        # Calculate the correlation coefficient matrix
        cors = (minMin + self.p * maxMax) / (sonCol + self.p * maxMax)
        # Seeking average comprehensive relevance
        meanCors = cors.mean(axis=0)
        self.result = {'cors': {'value': cors, 'desc': 'Correlation coefficient matrix'},
                       'meanCors': {'value': meanCors, 'desc': 'Average comprehensive correlation coefficient'}}


class GrayRelationalCoefficient():
    def __init__(self, data, tetha=0.5, standard=True):
        '''
        data: Input matrix, vertical axis is attribute name, first column is parent sequence
        theta: Resolution coefficient, range 0~1，Generally take 0.5，The smaller the correlation coefficient is, the greater the difference is, and the stronger the discrimination ability is
        standard: Need standardization
        '''
        self.data = np.array(data)
        self.tetha = tetha
        self.standard = standard

    def get_calculate_relational_coefficient(self, parent_column=0):
        self.normalize()
        return self._calculate_relational_coefficient(parent_column)

    def normalize(self):
        if not self.standard:
            return None

        self.scaler = StandardScaler().fit(self.data)
        self.data = self.scaler.transform(self.data)

    def _calculate_relational_coefficient(self, parent_column):
        momCol = self.data[:, parent_column].copy()
        sonCol = self.data[:, 0:]

        for col in range(sonCol.shape[1]):
            sonCol[:, col] = abs(sonCol[:, col] - momCol)

        minMin = np.nanmin(sonCol)
        maxMax = np.nanmax(sonCol)

        # Calculation of correlation coefficient matrix
        cors = (minMin + (self.tetha * maxMax)) / (sonCol + (self.tetha * maxMax))
        return cors


def plot_average_grey_relational_coefficient(mean_cors, columns, label='Label'):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    #Used to display negative sign normally
    plt.rcParams['axes.unicode_minus'] = False

    #Visualization matrix
    plt.clf()
    plt.figure(figsize=(8, 12))
    sns.heatmap(mean_cors.reshape(1, -1), square=True, annot=True,  cbar=False,
                vmax=1.0, linewidths=0.1, cmap='viridis')
    plt.yticks([0, ], [label])
    plt.xticks(np.arange(0.5, 3, 1), columns, rotation=90)
    plt.title('Index correlation matrix')