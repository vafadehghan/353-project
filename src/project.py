import sys
import pykalman as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import xml.etree.ElementTree as et
import math
from scipy import signal
from scipy.signal import find_peaks


def readfiles(files):
    df = pd.DataFrame()
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
    # print(df)

    # print(df[(df['position'] == 'Ankle') & (df['side'] == 'Left')])
    la = df[(df['position'] == 'Ankle') & (df['side'] == 'Left')] # 22 steps
    lax = la['aX'] 
    print(lax)    


    # b, a = signal.butter(2, 0.02, btype='lowpass', analog=False) The best one so far
    b, a = signal.butter(2, 0.02, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, lax) 

    peaks, _ = find_peaks(low_passed, height=0)

    # plt.plot(lax)     
    plt.plot(low_passed) 
    plt.plot(peaks, low_passed[peaks], "x")
    plt.show() 

    # plt.savefig('filtered.png')
if __name__ == '__main__':
    main()
