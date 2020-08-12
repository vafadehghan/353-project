import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import signal, stats 
from math import sqrt
from numpy.fft import fft, fftshift
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

def summarize(flag):
    if flag == "person":
        summaries_person = []
        
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
        
        df = pd.DataFrame(summaries_person , columns = ['class', 
        'mean(ax)' , 'std(ax)' , 'min(ax)' , '25th(ax)' , '50th(ax)' , '75th(ax)' , 'max(ax)',
        'mean(ay)' , 'std(ay)' , 'min(ay)' , '25th(ay)' , '50th(ay)' , '75th(ay)' , 'max(ay)',
        'mean(az)' , 'std(az)' , 'min(az)' , '25th(az)' , '50th(az)' , '75th(az)' , 'max(az)',
        'mean(aT)' , 'std(aT)' , 'min(aT)' , '25th(aT)' , '50th(aT)' , '75th(aT)' , 'max(aT)',
        'mean(freq)'])
        
        return df

    summaries_placement = []    
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
        
    

    df2 = pd.DataFrame(summaries_placement , columns = ['class', 
    'mean(ax)' , 'std(ax)' , 'min(ax)' , '25th(ax)' , '50th(ax)' , '75th(ax)' , 'max(ax)',
    'mean(ay)' , 'std(ay)' , 'min(ay)' , '25th(ay)' , '50th(ay)' , '75th(ay)' , 'max(ay)',
    'mean(az)' , 'std(az)' , 'min(az)' , '25th(az)' , '50th(az)' , '75th(az)' , 'max(az)',
    'mean(aT)' , 'std(aT)' , 'min(aT)' , '25th(aT)' , '50th(aT)' , '75th(aT)' , 'max(aT)',
    'mean(freq)'])
    return df2

def MLClassifier(X, y, n):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    bayes_walk_model = make_pipeline(
        StandardScaler(),
        GaussianNB()
    )
    kNN_walk_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors = 8)
    )
    neural_net_walk_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver = 'lbfgs', hidden_layer_sizes = (16,8,4), activation = 'logistic', max_iter=100000)
    )
    decisionTree_walk_model = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(max_depth = 125)
    )
    randomForest_walk_model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators = 1200 , max_depth = 5, min_samples_leaf = 8)
    )

    # Train the Classifiers
    bayes_train_avg = 0
    bayes_avg = 0
    kNN_train_avg = 0
    kNN_avg = 0
    neural_net_train_avg = 0
    neural_net_avg = 0
    decisionTree_train_avg = 0
    decisionTree_avg = 0
    randomForest_train_avg = 0
    randomForest_avg = 0 
    models = [bayes_walk_model, kNN_walk_model, neural_net_walk_model, decisionTree_walk_model, randomForest_walk_model]
    for index in range(n):
        for i, m in enumerate(models):
            m.fit(X_train, y_train)
        bayes_train_avg += bayes_walk_model.score(X_train, y_train)
        bayes_avg += bayes_walk_model.score(X_valid, y_valid)

        kNN_train_avg += kNN_walk_model.score(X_train , y_train)
        kNN_avg += kNN_walk_model.score(X_valid , y_valid)

        neural_net_train_avg += neural_net_walk_model.score(X_train , y_train)
        neural_net_avg += neural_net_walk_model.score(X_valid , y_valid)

        decisionTree_train_avg += decisionTree_walk_model.score(X_train , y_train)
        decisionTree_avg += decisionTree_walk_model.score(X_valid , y_valid)

        randomForest_train_avg += randomForest_walk_model.score(X_train , y_train)
        randomForest_avg += randomForest_walk_model.score(X_valid , y_valid)

    # See the validation score to see how well they are performing
    print ('''
    Bayesian Classifier's training average score: %3g
    Bayesian Classifier's validation average score: %3g
    kNN Classifier's training average score: %3g
    kNN Classifier's validation average score: %3g
    Neural Network Classifier's average training score: %3g
    Neural Network Classifier's average validation score: %3g
    Decision Tree's Classifier's average training score: %3g
    Decision Tree's Classifier's average validation score: %3g
    Random Forest's Classifier's average training score: %3g
    Random Forest's Classifier's average validation score: %3g
    ''' % (bayes_train_avg/n, bayes_avg/n,
    kNN_train_avg/n, kNN_avg/n,
    neural_net_train_avg/n, neural_net_avg/n,
    decisionTree_train_avg/n, decisionTree_avg/n,
    randomForest_train_avg/n, randomForest_avg/n))

def fourier_person_analysis():
    Song_in_pocket = fourier_transformation(read_csv('../data/pocket/pocket_' + str(1)))[0]
    Vafa_in_pocket = fourier_transformation(read_csv('../data/pocket/pocket_' + str(7)))[0]
    Song_on_hand = fourier_transformation(read_csv('../data/hand/hand_' + str(1)))[0]
    Vafa_on_hand = fourier_transformation(read_csv('../data/hand/hand_' + str(7)))[0]
    Song_on_ankle = fourier_transformation(read_csv('../data/ankle/ankle_' + str(1)))[0]
    Vafa_on_ankle = fourier_transformation(read_csv('../data/ankle/ankle_' + str(7)))[0]
    
    for i in range(2,13):
        if i <= 6:
            Song_in_pocket += fourier_transformation(read_csv('../data/pocket/pocket_' + str(i)))[0]
            Song_on_hand += fourier_transformation(read_csv('../data/hand/hand_' + str(i)))[0]
            Song_on_ankle += fourier_transformation(read_csv('../data/ankle/ankle_' + str(i)))[0]
        elif i > 7:
            Vafa_in_pocket += fourier_transformation(read_csv('../data/pocket/pocket_' + str(i)))[0]
            Vafa_on_hand += fourier_transformation(read_csv('../data/hand/hand_' + str(i)))[0]
            Vafa_on_ankle += fourier_transformation(read_csv('../data/ankle/ankle_' + str(i)))[0]

    dfs = [Song_in_pocket, Song_on_hand, Song_on_ankle, Vafa_in_pocket, Vafa_on_hand, Vafa_on_ankle]

    min_df = len(dfs[0].dropna())
    for i in range(len(dfs)):
        dfs[i]['aT'] = dfs[i]['aT'].dropna()
        dfs[i]['aT'] = dfs[i]['aT'] / 12
        dfs[i]['freq'] = dfs[i]['freq'].dropna()
        dfs[i]['freq'] = dfs[i]['freq']/ 12
        dfs[i] = dfs[i].dropna()

        if min_df > len(dfs[i]):
            min_df = len(dfs[i])

    for i in range(len(dfs)):
        dfs[i] = dfs[i][:min_df-1]

    anova_test = stats.f_oneway(dfs[0]['aT'], dfs[1]['aT'], dfs[2]['aT'], dfs[3]['aT'], dfs[4]['aT'], dfs[5]['aT'])
    print("Anova test's p-value = ", anova_test.pvalue)

    data = pd.DataFrame({'Vafa_in_pocket': dfs[0]['aT'], 'Song_on_hand': dfs[1]['aT'],
    'Vafa_on_ankle': dfs[2]['aT'], 'Song_in_pocket': dfs[3]['aT'], 'Vafa_on_hand': dfs[4]['aT'], 
    'Song_on_ankle': dfs[5]['aT']})

    melted = pd.melt(data)
    posthoc = pairwise_tukeyhsd(melted['value'], melted['variable'], alpha=0.05)

    return posthoc

def fourier_side_analysis_Song():
    Song_left_pocket = fourier_transformation(read_csv('../data/pocket/pocket_' + str(1)))[0]
    Song_right_pocket = fourier_transformation(read_csv('../data/pocket/pocket_' + str(2)))[0]
    Song_left_hand = fourier_transformation(read_csv('../data/hand/hand_' + str(1)))[0]
    Song_right_hand = fourier_transformation(read_csv('../data/hand/hand_' + str(2)))[0]
    Song_left_ankle = fourier_transformation(read_csv('../data/ankle/ankle_' + str(1)))[0]
    Song_right_ankle = fourier_transformation(read_csv('../data/ankle/ankle_' + str(2)))[0]
    
    for i in range(3,13):
        if i % 2 == 0:
            Song_right_pocket += fourier_transformation(read_csv('../data/pocket/pocket_' + str(i)))[0]
            Song_right_hand += fourier_transformation(read_csv('../data/hand/hand_' + str(i)))[0]
            Song_right_ankle += fourier_transformation(read_csv('../data/ankle/ankle_' + str(i)))[0]
        else:
            Song_left_pocket += fourier_transformation(read_csv('../data/pocket/pocket_' + str(i)))[0]
            Song_left_hand += fourier_transformation(read_csv('../data/hand/hand_' + str(i)))[0]
            Song_left_ankle += fourier_transformation(read_csv('../data/ankle/ankle_' + str(i)))[0]

    dfs = [Song_left_pocket, Song_right_pocket,  Song_left_hand, Song_right_hand , Song_left_ankle, Song_right_ankle]

    min_df = len(dfs[0].dropna())
    for i in range(len(dfs)):
        dfs[i]['aT'] = dfs[i]['aT'].dropna()
        dfs[i]['aT'] = dfs[i]['aT'] / 12
        dfs[i]['freq'] = dfs[i]['freq'].dropna()
        dfs[i]['freq'] = dfs[i]['freq']/ 12
        dfs[i] = dfs[i].dropna()

        if min_df > len(dfs[i]):
            min_df = len(dfs[i])

    for i in range(len(dfs)):
        dfs[i] = dfs[i][:min_df-1]

    anova_test = stats.f_oneway(dfs[0]['aT'], dfs[1]['aT'], dfs[2]['aT'], dfs[3]['aT'], dfs[4]['aT'], dfs[5]['aT'])
    print("Anova test's p-value = ", anova_test.pvalue)

    data = pd.DataFrame({'Song_left_pocket': dfs[0]['aT'], 'Song_right_pocket': dfs[1]['aT'],
    'Song_left_hand': dfs[2]['aT'], 'Song_right_hand': dfs[3]['aT'], 'Song_left_ankle': dfs[4]['aT'], 
    'Song_right_ankle': dfs[5]['aT']})

    melted = pd.melt(data)
    posthoc = pairwise_tukeyhsd(melted['value'], melted['variable'], alpha=0.05)

    return posthoc

def fourier_side_analysis_Vafa():
    Vafa_left_pocket = fourier_transformation(read_csv('../data/pocket/pocket_' + str(1)))[0]
    Vafa_right_pocket = fourier_transformation(read_csv('../data/pocket/pocket_' + str(2)))[0]
    Vafa_left_hand = fourier_transformation(read_csv('../data/hand/hand_' + str(1)))[0]
    Vafa_right_hand = fourier_transformation(read_csv('../data/hand/hand_' + str(2)))[0]
    Vafa_left_ankle = fourier_transformation(read_csv('../data/ankle/ankle_' + str(1)))[0]
    Vafa_right_ankle = fourier_transformation(read_csv('../data/ankle/ankle_' + str(2)))[0]
    
    for i in range(3,13):
        if i % 2 == 0:
            Vafa_left_pocket += fourier_transformation(read_csv('../data/pocket/pocket_' + str(i)))[0]
            Vafa_left_hand += fourier_transformation(read_csv('../data/hand/hand_' + str(i)))[0]
            Vafa_left_ankle += fourier_transformation(read_csv('../data/ankle/ankle_' + str(i)))[0]
        else:
            Vafa_right_pocket += fourier_transformation(read_csv('../data/pocket/pocket_' + str(i)))[0]
            Vafa_right_hand += fourier_transformation(read_csv('../data/hand/hand_' + str(i)))[0]
            Vafa_right_ankle += fourier_transformation(read_csv('../data/ankle/ankle_' + str(i)))[0]

    dfs = [Vafa_left_pocket, Vafa_right_pocket,  Vafa_left_hand, Vafa_right_hand ,Vafa_left_ankle ,  Vafa_right_ankle]

    min_df = len(dfs[0].dropna())
    for i in range(len(dfs)):
        dfs[i]['aT'] = dfs[i]['aT'].dropna()
        dfs[i]['aT'] = dfs[i]['aT'] / 12
        dfs[i]['freq'] = dfs[i]['freq'].dropna()
        dfs[i]['freq'] = dfs[i]['freq']/ 12
        dfs[i] = dfs[i].dropna()

        if min_df > len(dfs[i]):
            min_df = len(dfs[i])

    for i in range(len(dfs)):
        dfs[i] = dfs[i][:min_df-1]

    anova_test = stats.f_oneway(dfs[0]['aT'], dfs[1]['aT'], dfs[2]['aT'], dfs[3]['aT'], dfs[4]['aT'], dfs[5]['aT'])
    print("Anova test's p-value = ", anova_test.pvalue)

    data = pd.DataFrame({'Vafa_left_pocket': dfs[0]['aT'], 'Vafa_right_pocket': dfs[1]['aT'],
    'Vafa_left_hand': dfs[2]['aT'], 'Vafa_right_hand': dfs[3]['aT'], 'Vafa_left_ankle': dfs[4]['aT'], 
    'Vafa_right_ankle': dfs[5]['aT']})

    melted = pd.melt(data)
    posthoc = pairwise_tukeyhsd(melted['value'], melted['variable'], alpha=0.05)

    return posthoc

def fourier_placement_analysis():
    pocket = fourier_transformation(read_csv('../data/pocket/pocket_' + str(1)))[0]
    hand = fourier_transformation(read_csv('../data/hand/hand_' + str(1)))[0]
    ankle = fourier_transformation(read_csv('../data/ankle/ankle_' + str(1)))[0]

    
    for i in range(2,13):
        pocket += fourier_transformation(read_csv('../data/pocket/pocket_' + str(i)))[0]
        hand += fourier_transformation(read_csv('../data/hand/hand_' + str(i)))[0]
        ankle += fourier_transformation(read_csv('../data/ankle/ankle_' + str(i)))[0]
    dfs = [pocket, hand, ankle]

    min_df = len(dfs[0].dropna())
    for i in range(len(dfs)):
        dfs[i]['aT'] = dfs[i]['aT'].dropna()
        dfs[i]['aT'] = dfs[i]['aT'] / 12
        dfs[i]['freq'] = dfs[i]['freq'].dropna()
        dfs[i]['freq'] = dfs[i]['freq']/ 12
        dfs[i] = dfs[i].dropna()

        if min_df > len(dfs[i]):
            min_df = len(dfs[i])

    for i in range(len(dfs)):
        dfs[i] = dfs[i][:min_df-1]

    anova_test = stats.f_oneway(dfs[0]['aT'], dfs[1]['aT'], dfs[2]['aT'])
    print("Anova test's p-value = ", anova_test.pvalue)

    data = pd.DataFrame({'pocket': dfs[0]['aT'], 'hand': dfs[1]['aT'],
    'ankle': dfs[2]['aT']})

    melted = pd.melt(data)
    posthoc = pairwise_tukeyhsd(melted['value'], melted['variable'], alpha=0.05)

    return posthoc

def main():
    
    data_placement = summarize("placement")
    #print(data_placement)
    y = data_placement['class']
    X = data_placement.drop(columns='class', axis=1, inplace=False)
    print("MLClassifier on placement only: ")
    print("n = 1 (Score of one run): ")
    MLClassifier(X, y, 1)
    print("n = 50 (Average score of 50 runs): ")
    MLClassifier(X, y, 50)

    data_person = summarize("person")
    #print(data_person)
    y = data_person['class']
    X = data_person.drop(columns='class', axis=1, inplace=False)
    print("MLClassifier on placement & person only: ")
    print("n = 1 (Score of one run): ")
    MLClassifier(X, y, 1)
    print("n = 50 (Average score of 50 runs): ")
    MLClassifier(X, y, 50)
    print("-------------------------------------------")


    posthoc = fourier_placement_analysis()
    print(posthoc)
    posthoc.plot_simultaneous()
    plt.tight_layout()
    plt.savefig("Posthoc analysis (placement).png")
    posthoc1 = fourier_person_analysis()
    print(posthoc1)
    posthoc1.plot_simultaneous()
    plt.tight_layout()
    plt.savefig("Posthoc analysis (person).png")
    posthoc2 = fourier_side_analysis_Song()
    print(posthoc2)
    posthoc2.plot_simultaneous()
    plt.tight_layout()
    plt.savefig("Posthoc analysis (side-Song).png")
    posthoc3 = fourier_side_analysis_Vafa()
    print(posthoc3)
    posthoc3.plot_simultaneous()
    plt.tight_layout()
    plt.savefig("Posthoc analysis (side-Vafa).png")



if __name__ == "__main__":
    main()