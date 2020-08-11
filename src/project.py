import sys
import pykalman as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import xml.etree.ElementTree as et
import math

def readfiles(files):
    df = pd.DataFrame(
        columns=['aX', 'aY', 'aZ', 'aT', 'side', 'position', 'person'])
    for file in files:
        fileLoc = '../data/' + file
        temp = pd.read_csv(fileLoc, sep=',', header=0, names=[
                           'time', 'aX', 'aY', 'aZ', 'aT'])
        temp = temp.drop(['time'], axis=1)

        if file[2:3] == "L":
            temp['side'] = 'Left'
        else:
            temp['side'] = 'Right'

        if file[3:4] == "A":
            temp['position'] = 'Ankle'
        elif file[3:4] == "H":
            temp['position'] = 'Hand'
        elif file[3:4] == "P":
            temp['position'] = 'Pocket'

        if file[0:1] == "V":
            temp['person'] = 'Vafa'
        df = df.append(temp)

    return df

def main():
    files = ['V-LA-Lin.csv', 'V-RA-Lin.csv', 'V-LP-Lin.csv',
         'V-RP-Lin.csv', 'V-LH-Lin.csv', 'V-RH-Lin.csv']
    df = readfiles(files)
    print(df)


if __name__ == '__main__':
    main()
