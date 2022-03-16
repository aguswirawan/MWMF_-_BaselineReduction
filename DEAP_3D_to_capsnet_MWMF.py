import scipy.io as sio
import os
import glob
import pickle

if __name__ == '__main__':
    use_baseline = 'yes'
    label = 'all'
    if use_baseline=='yes':
        dataset_dir = '3D_dataset_MWMF_sub/with_base/'
    else:
        dataset_dir = '3D_dataset_MWMF_sub/without_base/'

    result_dir = f'deap_shuffled_MWMF_sub/{use_baseline}_{label}/'
    if os.path.isdir(result_dir)==False:
        os.makedirs(result_dir)

    for file in sorted(glob.glob(dataset_dir+'*.mat')):
        filename = file.split('\\')[-1].replace('DE_', '')
        print("processing: ",filename,"......")
        cnn_all = sio.loadmat(file)
        cnn_data = cnn_all['data']
        cnn_label_valence = cnn_all['valence_labels']
        cnn_label_arousal = cnn_all['arousal_labels']

        print(f'data shape: {cnn_data.shape}')
        print(f'label shape: {cnn_label_valence.shape}')
        with open(f'{result_dir}/{filename}_win_128_rnn_dataset.pkl', "wb") as fp:
            pickle.dump(cnn_data, fp, protocol=4)
        with open(f'{result_dir}/{filename}_win_128_labels_valence.pkl', "wb") as fp:
            pickle.dump(cnn_label_valence, fp)
        with open(f'{result_dir}/{filename}_win_128_labels_arousal.pkl', "wb") as fp:
            pickle.dump(cnn_label_arousal, fp)
