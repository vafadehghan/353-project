import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import signal, stats 
from math import sqrt
from numpy.fft import fft, fftshift

def read_csv(path):
    path = path + '.csv'
    df = pd.read_csv(path, header=0, names=['time', 'ax', 'ay', 'az'])                                                  
    df = df[(df['time'] >= 2) & (df['time'] <= 8)] # Limit the data size as not every data is of the same size
                                                   # Exclude the first 3 seconds 
    df['aT'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)  #Total acceleration   
    return df     

# Butterworth Filter
def BW_filter(data):
    b, a = signal.butter(2, 0.02, btype = 'lowpass', analog = False) # The best one so far
    low_passed = signal.filtfilt(b, a, data)
    
    # plt.plot(lax)
    # plt.plot(low_passed)
    # plt.plot(peaks, low_passed[peaks], "x")
    # plt.show()
    # plt.savefig('filtered.png')


    return low_passed

# First filter the data using Butterworth filter and using fft to transform the data
def fourier_transformation(data):
    # Filtering using Butterworth filter
    filtered = data.apply(BW_filter)
    filtered = filtered.reset_index(drop = True)
    
    # Using Fourier Transformation on the filtered data
    # fft = np.fft.fft(data)
    # print(fft)
    # plt.plot(fft)
    # plt.show()

    data = data.reset_index(drop=True)
    fourierT = filtered.apply(fft)      
    fourierT = fourierT.apply(fftshift)
    fourierT = fourierT.abs()
    
    # Calculating the sampling frequency and the average steps per second
    length = len(data)
    sampling_freq = round(length / data.at[length-1, 'time']) #samples per second
    fourierT['freq'] = np.linspace(-sampling_freq/2, sampling_freq/2, num = len(data))
    temp = fourierT[fourierT['freq'] > 0]
    peak_index = temp['aT'].nlargest(n = 1).idxmax()
    average = fourierT.at[peak_index , 'freq']
    peak_val_index = fourierT['aT'].nlargest(n = 1).idxmax()
    fourierT.at[peak_val_index, 'aT'] = temp['aT'].max()
    
    return fourierT , average

def calc_summary(data):
    # The parameter will be the original dataFrame after some data cleaning
    stats = []
    ax_summary = data['ax'].describe()      
    ay_summary = data['ay'].describe()
    az_summary = data['az'].describe()
    aT_summary = data['aT'].describe()
    for i in range(1 , 8):
        stats.append(ax_summary[i])  
    for i in range(1 , 8):
        stats.append(ay_summary[i])
    for i in range(1 , 8):
        stats.append(az_summary[i]) 
    for i in range(1 , 8):
        stats.append(aT_summary[i]) 
    return stats

def BW_stats(data):
    filtered = filtered = data.apply(BW_filter)
    stats = calc_summary(filtered)
    return stats

def summarize():
    summaries_person = []
    summaries_placement = []

         
    for i in range(1 , 13):
        df = read_csv('../data/pocket/pocket_' + str(i))
        summary = []
        if i <= 6:     
            summary.append('Song-Pocket')
        elif i > 6:
            summary.append('Vafa-Pocket')
        summary += BW_stats(df)
        summary.append(fourier_transformation(df)[1])
        summaries_person.append(summary)

    for i in range(1 , 13):
        df = read_csv('../data/hand/hand_' + str(i))
        summary = []
        if i <= 6:     
            summary.append('Song-Hand')
        else:
            summary.append('Vafa-Hand')
        summary += BW_stats(df)
        summary.append(fourier_transformation(df)[1])  
        summaries_person.append(summary)

    for i in range(1 , 13):
        df = read_csv('../data/ankle/ankle_' + str(i))
        summary = []
        if i <= 6:     
            summary.append('Song-Ankle')
        else:
            summary.append('Vafa-Ankle')
        summary += BW_stats(df)
        summary.append(fourier_transformation(df)[1])  
        summaries_person.append(summary)

    for i in range(1 , 13):
        df = read_csv('../data/pocket/pocket_' + str(i))
        summary = []
        summary.append('pocket')
        summary += BW_stats(df)
        summary.append(fourier_transformation(df)[1])     
        summaries_placement.append(summary)

    for i in range(1 , 13):
        df = read_csv('../data/hand/hand_' + str(i))
        summary = []
        summary.append('hand')
        summary += BW_stats(df)
        summary.append(fourier_transformation(df)[1])  
        summaries_placement.append(summary)

    for i in range(1 , 13):
        df = read_csv('../data/ankle/ankle_' + str(i))
        summary = []
        summary.append('ankle')
        summary += BW_stats(df)
        summary.append(fourier_transformation(df)[1])  
        summaries_placement.append(summary)
        
    df = pd.DataFrame(summaries_person , columns = ['class', 
    'mean(ax)' , 'std(ax)' , 'min(ax)' , '25th(ax)' , '50th(ax)' , '75th(ax)' , 'max(ax)',
    'mean(ay)' , 'std(ay)' , 'min(ay)' , '25th(ay)' , '50th(ay)' , '75th(ay)' , 'max(ay)',
    'mean(az)' , 'std(az)' , 'min(az)' , '25th(az)' , '50th(az)' , '75th(az)' , 'max(az)',
    'mean(aT)' , 'std(aT)' , 'min(aT)' , '25th(aT)' , '50th(aT)' , '75th(aT)' , 'max(aT)',
    'mean(freq)'])
    df.to_csv('summary_person.csv' , index = False)

    df2 = pd.DataFrame(summaries_placement , columns = ['class', 
    'mean(ax)' , 'std(ax)' , 'min(ax)' , '25th(ax)' , '50th(ax)' , '75th(ax)' , 'max(ax)',
    'mean(ay)' , 'std(ay)' , 'min(ay)' , '25th(ay)' , '50th(ay)' , '75th(ay)' , 'max(ay)',
    'mean(az)' , 'std(az)' , 'min(az)' , '25th(az)' , '50th(az)' , '75th(az)' , 'max(az)',
    'mean(aT)' , 'std(aT)' , 'min(aT)' , '25th(aT)' , '50th(aT)' , '75th(aT)' , 'max(aT)',
    'mean(freq)'])
    df2.to_csv('summary_placement.csv', index=False)

def main():
    summarize()

if __name__ == "__main__":
    main()