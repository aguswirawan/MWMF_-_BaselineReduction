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

def compute_DE(signal):
	variance = np.var(signal,ddof=1)
	return np.log(2*np.pi*np.e*variance)/2

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
    
# [0, respondent] ke-tiga, [fields] ke-lima, [baseline/stimuli] ke-tujuh, [trial, 0] ke-delapan

def baseline_mwmf(base_signal):
    # data_abs = np.abs(base_signal)
    # data_abs_min = np.min(data_abs)
    # data_abs_max = np.max(data_abs)
    data_mean = np.mean(base_signal)
    data_stdv = np.std(base_signal)
    # norm_func = lambda x: (x - data_abs_min) / (data_abs_max - data_abs_min)
    norm_func = lambda x: (x - data_mean) / data_stdv
    data_norm = norm_func(base_signal)
    data_norm_abs = np.abs(data_norm)
    data_pad_norm = np.append(np.append([0], data_norm_abs), [0])
    data_pad_base = np.append(np.append([0], base_signal), [0])

    results = []
    i = 0
    j = 3
    while j <= len(data_pad_norm):
        # print(i)
        data_base_tmp = data_pad_base[i:j]
        data_norm_tmp = data_pad_norm[i:j]
        sum_norm_tmp = np.sum(data_norm_tmp)
        mwmf_func = lambda x, y: x / sum_norm_tmp * y
        result = mwmf_func(data_norm_tmp, data_base_tmp)
        results.append(np.mean(result))
        i += 1
        j += 1

    return np.array(results)

def decompose(data):
  # trial*channel*sample
  start_index = 512 #4s pre-trial signals
  # data = data_dreamer['DREAMER'].copy()
  frequency = 128

  decomposed_de = np.empty([0,4,])

  base_DE = np.empty([0,56])

  data_seconds_list = np.empty([0])

  for trial in range(18):
    temp_base_DE = np.empty([0])
    temp_base_theta_DE = np.empty([0])
    temp_base_alpha_DE = np.empty([0])
    temp_base_beta_DE = np.empty([0])
    temp_base_gamma_DE = np.empty([0])
    
    # print(data.shape)
    data_seconds = int(data[0, 0][2][0, 0][1][trial, 0].shape[0]/frequency)

    temp_de = np.empty([0,data_seconds])

    for channel in range(14):
      
      trial_signal = data[0, 0][2][0, 0][1][trial, 0][:, channel]
      base_signal = data[0, 0][2][0, 0][0][trial, 0][:-640, channel]
      
      # # mav
      base_signal = baseline_mwmf(base_signal)
            #base_signal = baseline_mav(base_signal, is_mav_2=True) # mav_2
            #base_signal = baseline_mav(base_signal, is_mav_2=False) # mav_1
            
      #****************compute base DE****************
      base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
      base_alpha = butter_bandpass_filter(base_signal, 8,14, frequency, order=3)
      base_beta = butter_bandpass_filter(base_signal,14,31, frequency, order=3)
      base_gamma = butter_bandpass_filter(base_signal,31,45, frequency, order=3)

      base_theta_DE = (compute_DE(base_theta[:128])+compute_DE(base_theta[128:256])+compute_DE(base_theta[256:384])+compute_DE(base_theta[384:512])+compute_DE(base_theta[512:]))/5
      base_alpha_DE =(compute_DE(base_alpha[:128])+compute_DE(base_alpha[128:256])+compute_DE(base_alpha[256:384])+compute_DE(base_alpha[384:512])+compute_DE(base_alpha[512:]))/5
      base_beta_DE =(compute_DE(base_beta[:128])+compute_DE(base_beta[128:256])+compute_DE(base_beta[256:384])+compute_DE(base_beta[384:512])+compute_DE(base_beta[512:]))/5
      base_gamma_DE =(compute_DE(base_gamma[:128])+compute_DE(base_gamma[128:256])+compute_DE(base_gamma[256:384])+compute_DE(base_gamma[384:512])+compute_DE(base_gamma[512:]))/5

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
    # print(base_DE.shape)
    # print(temp_base_DE.shape)
    base_DE = np.vstack([base_DE,temp_base_DE])

    data_seconds_list = np.append(data_seconds_list, data_seconds)

  # decomposed_de = decomposed_de.reshape(-1,14,4,57).transpose([0,3,2,1]).reshape(-1,4,14).reshape(-1,56)
  decomposed_de = decomposed_de.reshape(-1,14,4).transpose([0,2,1]).reshape(-1,4,14).reshape(-1,56)
  print("base_DE shape:",base_DE.shape)
  print("trial_DE shape:",decomposed_de.shape)
  return base_DE, decomposed_de, data_seconds_list

def get_labels(data):
  
  # data = data_dreamer['DREAMER'][0, 0][0][0, 0][0, 0].copy()

  valence_labels = data[0, 0][4]>=3	# valence labels
  arousal_labels = data[0, 0][5]>=3	# arousal labels
  dominance_labels = data[0, 0][6]>=3 # dominance labels

  final_valence_labels = np.empty([0])
  final_arousal_labels = np.empty([0])
  final_dominance_labels = np.empty([0])

  for i in range(len(valence_labels)):
    data_seconds = int(data[0, 0][2][0, 0][1][i, 0].shape[0]/128)
    for j in range(0,data_seconds):
      final_valence_labels = np.append(final_valence_labels,valence_labels[i])
      final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])
      final_dominance_labels = np.append(final_dominance_labels,dominance_labels[i])
  print("labels:",final_arousal_labels.shape)
  return final_arousal_labels, final_valence_labels, final_dominance_labels

if __name__ == '__main__':
  dataset_dir = "../DREAMER/"

  result_dir = "1D_dataset_MWMF/"
  if os.path.isdir(result_dir)==False:
    os.makedirs(result_dir)

  data_dreamer =  sio.loadmat(dataset_dir+'DREAMER.mat')

  for respondent in range(23):
    print("processing respondent: ", respondent+1, "......")
    data_respondent = data_dreamer['DREAMER'][0, 0][0][0, respondent].copy()
    base_DE, trial_DE, data_seconds_list = decompose(data_respondent)
    # print(trial_DE[:10, :14])
    arousal_labels, valence_labels, dominance_labels = get_labels(data_respondent)
    sio.savemat(result_dir+"DE_res"+str(respondent+1).zfill(2)+".mat",{"base_data":base_DE,
                                                "data":trial_DE,
                                                "valence_labels":valence_labels,
                                                "arousal_labels":arousal_labels,
                                                "dominance_labels":dominance_labels,
                                                "data_seconds_list":data_seconds_list})