import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import scipy.io as sio
import numpy as np
import os
import math
import sys
import glob

def read_file(file):
    file = sio.loadmat(file)
    trial_data = file['data']
    base_data = file["base_data"]
    data_seconds_list = file['data_seconds_list']
    return trial_data,base_data,file["arousal_labels"],file["valence_labels"],file['dominance_labels'],data_seconds_list

def get_vector_deviation(vector1,vector2):
	return (vector1/vector2)

def get_dataset_deviation(trial_data,base_data,data_seconds_list):
    new_dataset = np.empty([0,56])
    second_now = 0
    for i, seconds in enumerate(data_seconds_list[0]):
        for j in range(int(seconds)):
            new_record = get_vector_deviation(trial_data[j+second_now], base_data[i]).reshape(1,56)
            new_dataset = np.vstack([new_dataset,new_record])
    second_now += int(seconds)
    return new_dataset

def data_1Dto2D(data, Y=9, X=9):
	data_2D = np.zeros([Y, X])
	data_2D[0] = (0,        0,          0,          0,          0,          0,          0,          0,          0       )
	data_2D[1] = (0,        0,          0,          data[0],    0,          data[13],   0,          0,          0       )
	data_2D[2] = (data[1],  0,          data[2],    0,          0,          0,          data[11],   0,          data[12])
	data_2D[3] = (0,        data[3],    0,          0,          0,          0,          0,          data[10],   0       )
	data_2D[4] = (data[4],  0,          0,          0,          0,          0,          0,          0,          data[9] )
	data_2D[5] = (0,        0,          0,          0,          0,          0,          0,          0,          0       )
	data_2D[6] = (data[5],  0,          0,          0,          0,          0,          0,          0,          data[8] )
	data_2D[7] = (0,        0,          0,          0,          0,          0,          0,          0,          0       )
	data_2D[8] = (0,        0,          0,          data[6],    0,          data[7],    0,          0,          0       )
	# return shape:9*9
	return data_2D

def pre_process(path,y_n):
	# DE feature vector dimension of each band
	data_3D = np.empty([0,9,9])
	sub_vector_len = 14
	trial_data,base_data,arousal_labels,valence_labels,dominance_labels,data_seconds_list = read_file(path)
	if y_n=="yes":
		data = get_dataset_deviation(trial_data,base_data,data_seconds_list)
		data = preprocessing.scale(data,axis=1, with_mean=True,with_std=True,copy=True)
	else:
		data = preprocessing.scale(trial_data,axis=1, with_mean=True,with_std=True,copy=True)
	# convert 128 vector ---> 4*9*9 cube
	for vector in data:
		for band in range(0,4):
			data_2D_temp = data_1Dto2D(vector[band*sub_vector_len:(band+1)*sub_vector_len])
			data_2D_temp = data_2D_temp.reshape(1,9,9)
			# print("data_2d_temp shape:",data_2D_temp.shape)
			data_3D = np.vstack([data_3D,data_2D_temp])
	data_3D = data_3D.reshape(-1,4,9,9)
	print("final data shape:",data_3D.shape)
	return data_3D,arousal_labels,valence_labels,dominance_labels

if __name__ == '__main__':
	dataset_dir = "1D_dataset_MWMF/Group/"
	use_baseline = 'yes' # yes or no
	if use_baseline=="yes":
		result_dir = "3D_dataset_MWMF_div/Group/with_base/"
		if os.path.isdir(result_dir)==False:
			os.makedirs(result_dir)
	else:
		result_dir = "3D_dataset_default_div/Group/without_base/"
		if os.path.isdir(result_dir)==False:
			os.makedirs(result_dir)

	for file in sorted(glob.glob(dataset_dir+'*.mat')):
		filename = file.split('\\')[-1]
		print("processing: ",filename,"......")
		data,arousal_labels,valence_labels,dominance_labels = pre_process(file,use_baseline)
		print("final shape:",data.shape)
		sio.savemat(result_dir+filename,{"data":data,"valence_labels":valence_labels,"arousal_labels":arousal_labels,"dominance_labels":dominance_labels})
