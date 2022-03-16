import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from scipy.stats import norm
import glob

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def read_file(file):
	data = sio.loadmat(file)
	# data = data['data']
	# print(data.shape)
	return data

def compute_DE(signal):
	variance = np.var(signal,ddof=1)
	return math.log(2*math.pi*math.e*variance)/2

# def baseline_mav(base_signal, is_mav_2=False):
# 	df_base_signal = pd.DataFrame({'base_signal': base_signal})
# 	df_base_signal['abs_base_signal'] = np.abs(base_signal)
# 	min_abs_base_signal = df_base_signal['abs_base_signal'].min()
# 	df_base_signal['bobot'] = df_base_signal['abs_base_signal'].apply(lambda x: np.abs(1 - (x - min_abs_base_signal)))

# 	i = 0
# 	j = 3
# 	mav_1 = df_base_signal['base_signal'].values.tolist()[:2]
# 	while j <= len(df_base_signal):
# 		norm_bobot = df_base_signal['bobot'].iloc[i:j].sum()
# 		mav = (df_base_signal['base_signal'].iloc[i:j] * df_base_signal['bobot'].iloc[i:j] / norm_bobot).mean()
# 		mav_1.append(mav)
# 		i += 1
# 		j += 1

# 	df_base_signal['mav_1'] = mav_1
# 	if not is_mav_2:
# 		base_signal = df_base_signal['mav_1'].values
# 		return base_signal

# 	i = 0
# 	j = 3
# 	mav_2 = []
# 	while j <= len(df_base_signal):
# 		norm_bobot = df_base_signal['bobot'].iloc[i:j].sum()
# 		mav = (df_base_signal['mav_1'].iloc[i:j] * df_base_signal['bobot'].iloc[i:j] / norm_bobot).mean()
# 		mav_2.append(mav)
# 		i += 1
# 		j += 1
# 	mav_2.extend(df_base_signal['mav_1'].values.tolist()[-2:])

# 	df_base_signal['mav_2'] = mav_2
# 	if is_mav_2:
# 		base_signal = df_base_signal['mav_2'].values
# 		return base_signal

def decompose(data):
    # trial*channel*sample
    # data = read_file(file)
    #shape = data.shape
    frequency = 128
    decomposed_de = np.empty([0,4,])
    base_DE = np.empty([0,56])
    data_seconds_list = np.empty([0])
    for trial in range(16):
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0])
        temp_base_alpha_DE = np.empty([0])
        temp_base_beta_DE = np.empty([0])
        temp_base_gamma_DE = np.empty([0])
        
        data_seconds = (data[:, trial][0].shape[0]//frequency) + 1 - 5
        temp_de = np.empty([0,data_seconds])
        
        for channel in range(14):
            
            #The final 3 seconds of the baseline signal
            trial_signal = data[:, trial][0][640:, channel]
            base_signal = data[:, trial][0][:640, channel]

            frequency_remainder = trial_signal.shape[0] % 128
            if  frequency_remainder != 0:
                trial_signal = np.append(trial_signal, np.zeros(frequency-frequency_remainder))

			# # mav
            #base_signal = baseline_mav(base_signal, is_mav_2=True) # mav_2
            #base_signal = baseline_mav(base_signal, is_mav_2=False) # mav_1

			#****************compute base DE****************
            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8,14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal,14,31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal,31,45, frequency, order=3)
            
            base_theta_DE = (compute_DE(base_theta[:128])+compute_DE(base_theta[128:256])+compute_DE(base_theta[256:])+compute_DE(base_theta[384:])+compute_DE(base_theta[512:]))/5
            base_alpha_DE =(compute_DE(base_alpha[:128])+compute_DE(base_alpha[128:256])+compute_DE(base_alpha[256:])+compute_DE(base_alpha[384:])+compute_DE(base_alpha[512:]))/5
            base_beta_DE =(compute_DE(base_beta[:128])+compute_DE(base_beta[128:256])+compute_DE(base_beta[256:])+compute_DE(base_beta[384:])+compute_DE(base_beta[512:]))/5
            base_gamma_DE =(compute_DE(base_gamma[:128])+compute_DE(base_gamma[128:256])+compute_DE(base_gamma[256:])+compute_DE(base_gamma[384:])+compute_DE(base_gamma[512:]))/5
            
            temp_base_theta_DE = np.append(temp_base_theta_DE,base_theta_DE)
            temp_base_gamma_DE = np.append(temp_base_gamma_DE,base_gamma_DE)
            temp_base_beta_DE = np.append(temp_base_beta_DE,base_beta_DE)
            temp_base_alpha_DE = np.append(temp_base_alpha_DE,base_alpha_DE)
            
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)
            
            DE_theta = np.zeros(shape=[0],dtype = float)
            DE_alpha = np.zeros(shape=[0],dtype = float)
            DE_beta =  np.zeros(shape=[0],dtype = float)
            DE_gamma = np.zeros(shape=[0],dtype = float)
            
            for index in range(data_seconds):
                DE_theta =np.append(DE_theta,compute_DE(theta[index*frequency:(index+1)*frequency]))
                DE_alpha =np.append(DE_alpha,compute_DE(alpha[index*frequency:(index+1)*frequency]))
                DE_beta =np.append(DE_beta,compute_DE(beta[index*frequency:(index+1)*frequency]))
                DE_gamma =np.append(DE_gamma,compute_DE(gamma[index*frequency:(index+1)*frequency]))
            temp_de = np.vstack([temp_de,DE_theta])
            temp_de = np.vstack([temp_de,DE_alpha])
            temp_de = np.vstack([temp_de,DE_beta])
            temp_de = np.vstack([temp_de,DE_gamma])
        temp_trial_de = temp_de.reshape(-1,4,)
        decomposed_de = np.vstack([decomposed_de,temp_trial_de])
        
        temp_base_DE = np.append(temp_base_theta_DE,temp_base_alpha_DE)
        temp_base_DE = np.append(temp_base_DE,temp_base_beta_DE)
        temp_base_DE = np.append(temp_base_DE,temp_base_gamma_DE)
        base_DE = np.vstack([base_DE,temp_base_DE])
        data_seconds_list = np.append(data_seconds_list, data_seconds)
    decomposed_de = decomposed_de.reshape(-1,14,4).transpose([0,2,1]).reshape(-1,4,14).reshape(-1,56)
    print("base_DE shape:",base_DE.shape)
    print("trial_DE shape:",decomposed_de.shape)
    return base_DE, decomposed_de, data_seconds_list

def get_labels(data_labels, data_trial):
    	#0 valence, 1 arousal, 2 dominance, 3 liking

    #valence_labels = data[:,0]>5	# valence labels
    #arousal_labels = data[:,1]>5	# arousal labels
    #dominance_labels = data[:,2]>5 # dominance labels
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    final_dominance_labels = np.empty([0])
    for trial in range(16):
        
        data_label = data_labels[:, trial][0][0].copy()
        data_trial_row = (data_trial[:, trial][0].shape[0]//128) + 1 - 5  

        arousal_labels = np.array([data_label[0]>5] * data_trial_row)
        valence_labels = np.array([data_label[1]>5] * data_trial_row)
        dominance_labels = np.array([data_label[2]>5] * data_trial_row)

        final_valence_labels = np.append(final_valence_labels,valence_labels)
        final_arousal_labels = np.append(final_arousal_labels,arousal_labels)
        final_dominance_labels = np.append(final_dominance_labels,dominance_labels)
        

    print("labels_valence:",final_valence_labels.shape)
    print("labels_arousal:",final_arousal_labels.shape)
    print("labels_dominance:",final_dominance_labels.shape)
    return final_arousal_labels,final_valence_labels,final_dominance_labels

# def wgn(x, snr):
#     snr = 10**(snr/10.0)
#     xpower = np.sum(x**2)/len(x)
#     npower = xpower / snr
#     return np.random.randn(len(x)) * np.sqrt(npower)

# def feature_normalize(data):
# 	mean = data[data.nonzero()].mean()
# 	sigma = data[data. nonzero ()].std()
# 	data_normalized = data
# 	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
# 	return data_normalized


if __name__ == '__main__':
    dataset_dir = "../Amigos/Individual/"
    result_dir = "1D_dataset_default/Individual/"
    if os.path.isdir(result_dir)==False:
         os.makedirs(result_dir)
    for file in sorted(glob.glob(dataset_dir+'*.mat')):
        filename = file.split('\\')[-1]
        print("processing: ",filename,"......")
        file_path = os.path.join(dataset_dir,filename)
        data = read_file(file)
        base_DE, trial_DE, data_seconds_list = decompose(data['joined_data'].copy())
        arousal_labels, valence_labels, dominance_labels = get_labels(data['labels_selfassessment'].copy(), data['joined_data'].copy())
        sio.savemat(result_dir+"DE_"+filename,{"base_data":base_DE,
                                         "data":trial_DE,
                                         "valence_labels":valence_labels,
                                         "arousal_labels":arousal_labels,
                                         "dominance_labels":dominance_labels,
                                         "data_seconds_list":data_seconds_list})