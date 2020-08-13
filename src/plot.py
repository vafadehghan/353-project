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
    df = pd.DataFrame(
        columns=['aX', 'aY', 'aZ', 'aT', 'velocity', 'position', 'side', 'placement', 'person'])
    for file in files:
        fileLoc = '../data/' + file
        temp = pd.read_csv(fileLoc, sep=',', header=0, names=[
                           'time', 'aX', 'aY', 'aZ', 'aT'])
        # temp = temp.drop(['time'], axis=1)
        temp['time_shift'] = temp['time'].shift(1)
        temp['velocity'] = (temp['time'] - temp['time_shift']) * temp['aX']

        temp['position'] = temp['velocity'] * \
            (temp['time'] - temp['time_shift'])
        temp = temp.drop(['time'], axis=1)
        temp = temp.drop(['time_shift'], axis=1)

        if file[2:3] == "L":
            temp['side'] = 'Left'
        else:
            temp['side'] = 'Right'

        if file[3:4] == "A":
            temp['placement'] = 'Ankle'
        elif file[3:4] == "H":
            temp['placement'] = 'Hand'
        elif file[3:4] == "P":
            temp['placement'] = 'Pocket'
        if file[0:1] == "V":
            temp['person'] = 'Vafa'
        df = df.append(temp)
    df = df.dropna()
    return df


def main():
    files = ['V-LA-Lin.csv', 'V-RA-Lin.csv', 'V-LP-Lin.csv',
             'V-RP-Lin.csv', 'V-LH-Lin.csv', 'V-RH-Lin.csv']
    df = readfiles(files)
    # print(df)

    # print(df[(df['placement'] == 'Ankle') & (df['side'] == 'Right')])
    la = df[(df['placement'] == 'Ankle') & (df['side'] == 'Right')]  # 22 steps
    lax = la['aX']
    # print(la)

    b, a = signal.butter(2, 0.02, btype='lowpass', analog=False)
    # b, a = signal.butter(2, 0.02, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, lax)

    # peaks, _ = find_peaks(low_passed, height=0)

    # fft = np.fft.fft(low_passed)
    # print(fft)
    # plt.plot(fft)
    # plt.show()


    # plt.plot(lax)
    plt.plot(low_passed)
    # plt.plot(peaks, low_passed[peaks], "x")
    
    
    plt.xlabel("Time")
    plt.ylabel("$m/s^2$")
    plt.title("Right Ankle Filtered Accelerometer Data (X Axis)")
    # plt.show()
    plt.savefig('filtered.png')
if __name__ == '__main__':
    main()
