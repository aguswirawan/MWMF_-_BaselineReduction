"""
Keras implementation of Multi-level Features Guided Capsule Network (MLF-CapsNet).
This file trains a MLF-CapsNet on DEAP/DREAMER dataset with the parameters as mentioned in paper.
We have developed this code using the following GitHub repositories:
- Xifeng Guo's CapsNet code (https://github.com/XifengGuo/CapsNet-Keras)

Usage:
       python capsulenet-multi-gpu.py --gpus 2

"""

from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers,regularizers


K.set_image_data_format('channels_last')

import pandas as pd
import time
import pickle
import numpy as np


def data_load(data_file,dimention,debaseline):
    rnn_suffix = ".mat_win_128_rnn_dataset.pkl"
    label_suffix_valence = ".mat_win_128_labels_valence.pkl"
    label_suffix_arousal = ".mat_win_128_labels_arousal.pkl"
    arousal_or_valence = dimention
    with_or_without = debaseline # 'yes','not'
    dataset_dir = "amigos_shuffled_MWMF_sub/Group/"+ with_or_without + "_" + arousal_or_valence + "/"

    ###load training set
    with open(dataset_dir + data_file + rnn_suffix, "rb") as fp:
        rnn_datasets = pickle.load(fp)
    with open(dataset_dir + data_file + label_suffix_valence, "rb") as fp:
        label_valence = pickle.load(fp)
        # labels = pickle.load(fp)
        # labels = np.transpose(labels[0])
    with open(dataset_dir + data_file + label_suffix_arousal, 'rb') as fp:
        label_arousal = pickle.load(fp)

    def preprocess_two_labels(label_a, label_b):
        if label_a == 1 and label_b == 1:
            return 'HAHV'
        elif label_a == 0 and label_b == 1:
            return 'LAHV'
        elif label_a == 1 and label_b == 0:
            return 'HALV'
        elif label_a == 0 and label_b == 0:
            return 'LALV'

    labels = list(map(preprocess_two_labels, label_arousal.tolist()[0], label_valence.tolist()[0]))
    # print(labels[:5])

    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    print(encoder.classes_)
    # convert integers to dummy variables (i.e. one hot encoded)
    # print(encoded_labels[:5])
    labels = np_utils.to_categorical(encoded_labels)
    # print(labels[:5, :])

    # labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)


    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    rnn_datasets = rnn_datasets[index]  # .transpose(0,2,1)
    labels = labels[index]

    print(rnn_datasets.shape)
    datasets = rnn_datasets.reshape(-1, 9, 9, 4).astype('float32')
    labels = labels.astype('float32')
    print(datasets.shape)
    print(labels.shape)

    return datasets , labels


def Convolution(input_shape, n_class):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)
    # print(x.shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='same', activation='relu', name='conv1')(x)
    # print(conv1.shape)
    conv2 = layers.Conv2D(filters=128, kernel_size=4, strides=1, padding='same', activation='relu', name='conv2')(conv1)
    conv3 = layers.Conv2D(filters=256, kernel_size=4, strides=1, padding='same', activation='relu', name='conv3')(conv2)
    conv4 = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='conv4')(conv3)
    out_flat = layers.Flatten()(conv4)
    # conv5 = layers.Conv2D(filters=1024, kernel_size=1, strides=1, padding='same', activation='selu', name='conv5')(conv4)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    # primarycaps = PrimaryCap(conv5, dim_capsule=8, n_channels=32, kernel_size=4, strides=2, padding='same')
    # primarycaps = layers.Reshape(target_shape=[-1, 8], name='primarycap_reshape')(conv3)
    # print('primarycaps shape:')
    # print(primarycaps.shape)

    # Layer 3: Capsule layer. Routing algorithm works here.
    # digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    # out_caps = Length(name='capsnet')(digitcaps)

    fcn = models.Sequential()
    fcn.add(layers.Dense(1024, activation='relu'))
    fcn.add(layers.Dropout(0.5))
    fcn.add(layers.Dense(4, activation='softmax'))

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    # masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    # masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    # decoder = models.Sequential(name='decoder')
    # decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    # decoder.add(layers.Dense(1024, activation='relu'))
    # decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    # decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    # train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    # eval_model = models.Model(x, [out_caps, decoder(masked)])
    # train_model = models.Model([x, y], out_caps)
    # eval_model = models.Model(x, out_caps)

    train_model = models.Model([x, y], fcn(out_flat))
    eval_model = models.Model(x, fcn(out_flat))

    # manipulate model
    # noise = layers.Input(shape=(n_class, 16))
    # noised_digitcaps = layers.Add()([digitcaps, noise])
    # masked_noised_y = Mask()([noised_digitcaps, y])
    # manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    # return train_model, eval_model, manipulate_model
    return train_model, eval_model

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def train(model,  # type: models.Model
          data, args, fold):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + f'/log_fold_{fold}.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_CNN_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    # model.compile(optimizer=optimizers.Adam(lr=args.lr),
    #               loss=[margin_loss, 'mse'],
    #               loss_weights=[1., args.lam_recon],
    #               metrics={'capsnet': 'accuracy'})
    # model.compile(optimizer=optimizers.Adam(lr=args.lr),
    #               loss=margin_loss,
    #               # loss='categorical_crossentropy',
    #               metrics={'capsnet': 'accuracy'},
    #               # metrics='accuracy'
    #               )
    model.compile(optimizer=optimizers.RMSprop(lr=args.lr, momentum=0.09),
                  loss=margin_loss,
                  # loss='categorical_crossentropy',
                  # metrics={'capsnet': 'accuracy'},
                  metrics='accuracy'
                  )

    # Training without data augmentation:
    history = model.fit([x_train, y_train], y_train, batch_size=args.batch_size, epochs=args.epochs,
                        callbacks=[log, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    # from utils import plot_log
    # plot_log(args.save_dir + '/log.csv', show=True)

    return model, history

time_start_whole = time.time()

dataset_name = 'amigos' #'deap' # dreamer

# subjects = ['Data_Preprocessed_P05','Data_Preprocessed_P30'] 
# subjects = ['Data_Preprocessed_P04','Data_Preprocessed_P10','Data_Preprocessed_P25'] 
# subjects = ['Data_Preprocessed_P01','Data_Preprocessed_P02','Data_Preprocessed_P03','Data_Preprocessed_P06','Data_Preprocessed_P07','Data_Preprocessed_P09','Data_Preprocessed_P13','Data_Preprocessed_P14','Data_Preprocessed_P15', 'Data_Preprocessed_P16','Data_Preprocessed_P19', 'Data_Preprocessed_P20'] 
subjects = ['Data_Preprocessed_P26','Data_Preprocessed_P27','Data_Preprocessed_P29','Data_Preprocessed_P31','Data_Preprocessed_P34','Data_Preprocessed_P35','Data_Preprocessed_P36','Data_Preprocessed_P37', 'Data_Preprocessed_P38','Data_Preprocessed_P39', 'Data_Preprocessed_P40'] 
# subjects = ['Data_Preprocessed_P08','Data_Preprocessed_P28','Data_Preprocessed_P32'] 

dimentions = ['all']#,'arousal','dominance']
debaseline = 'yes' # yes or not
tune_overfit = 'tune_overfit'
model_version = 'v0' # v0:'CapsNet', v1:'MLF-CapsNet(w/o)', v2:'MLF-CapsNet'


if __name__ == "__main__":
    for dimention in dimentions:
        for subject in subjects:
            import numpy as np
            import tensorflow as tf
            import os
            from tensorflow.keras import callbacks
            #from tensorflow.keras.utils.vis_utils import plot_model
            #from keras.utils import multi_gpu_model

            # setting the hyper parameters
            import argparse
            parser = argparse.ArgumentParser(description="Convolution Neural Network on " + dataset_name)
            parser.add_argument('--epochs', default=50, type=int)  # v0:20, v2:40
            parser.add_argument('--batch_size', default=120, type=int)
            parser.add_argument('--lam_regularize', default=0.0, type=float,
                                help="The coefficient for the regularizers")
            parser.add_argument('--debug', default=0, type=int,
                                help="Save weights by TensorBoard")
            parser.add_argument('--save_dir', default='result_MWMF_sub/Group/sub_dependent_'+ model_version +'/') # other
            parser.add_argument('-t', '--testing', action='store_true',
                                help="Test the trained model on testing dataset")
            parser.add_argument('-w', '--weights', default=None,
                                help="The path of the saved weights. Should be specified when testing")
            parser.add_argument('--lr', default=1e-4, type=float,
                                help="Initial learning rate")  # v0:0.0001, v2:0.00001
            parser.add_argument('--gpus', default=0, type=int)
            parser.add_argument('--lam_recon', default=0.392, type=float,
                                help="The coefficient for the loss of decoder")
            parser.add_argument('--lr_decay', default=0.9, type=float,
                                help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
            # parser.add_argument('--fold', default=10, type=int,
                                # help="K fold cross validation")
            args = parser.parse_args()

            print(time.asctime(time.localtime(time.time())))
            print(args)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            if dataset_name == 'amigos':          # load dreamer data
                datasets,labels = data_load(subject,dimention,debaseline)
           
                

            args.save_dir = args.save_dir + '/' + debaseline + '/' + subject + '_' + dimention + str(args.epochs)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            fold = 10
            test_accuracy_allfold = np.zeros(shape=[0], dtype=float)
            train_used_time_allfold = np.zeros(shape=[0], dtype=float)
            test_used_time_allfold = np.zeros(shape=[0], dtype=float)
            loss_allfold = []
            for curr_fold in range(fold):
                fold_size = datasets.shape[0] // fold
                indexes_list = [i for i in range(len(datasets))]
                #indexes = np.array(indexes_list)
                split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
                # print(split_list)
                split = np.array(split_list)
                # print(split)
                x_test = datasets[split]
                y_test = labels[split]

                split = np.array(list(set(indexes_list) ^ set(split_list)))
                x_train = datasets[split]
                y_train = labels[split]

                train_sample = y_train.shape[0]
                print("training examples:", train_sample)
                test_sample = y_test.shape[0]
                print("test examples    :", test_sample)
                print(x_train.shape, x_test.shape)
                print(y_train.shape, y_test.shape)

                # define model
                model, eval_model = Convolution(input_shape=x_train.shape[1:],
                                            n_class=len(np.unique(np.argmax(y_train, 1))))
                model.summary()
                # plot_model(model, to_file=args.save_dir+'/model_fold'+str(curr_fold)+'.png', show_shapes=True)

                # define muti-gpu model
                # multi_model = multi_gpu_model(model, gpus=args.gpus)
                multi_model = model
                # train
                train_start_time = time.time()
                _, history = train(model=multi_model, data=((x_train, y_train), (x_test, y_test)), args=args, fold=curr_fold)
                train_used_time_fold = time.time() - train_start_time
                # model.save_weights(args.save_dir + '/trained_model_fold'+str(curr_fold)+'.h5')
                print('Trained model saved to \'%s/trained_model_fold%s.h5\'' % (args.save_dir,curr_fold))
                print('Train time: ', train_used_time_fold)

                #test
                print('-' * 30 + 'fold  ' + str(curr_fold) + '  Begin: test' + '-' * 30)
                test_start_time = time.time()
                y_pred = eval_model.predict(x_test, batch_size=args.batch_size)  # batch_size = 100
                test_used_time_fold = time.time() - test_start_time
                # print(y_pred)
                # print(y_test)
                # print(y_pred.shape)
                # print(y_test.shape)
                # print(type(y_pred))
                # print(type(y_test))
                # print(len(y_pred))
                # print(y_test.shape)
                test_acc_fold = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
                #print('shape of y_pred: ',y_pred.shape[0])
                #print('y_pred: ', y_pred)
                #print('y_test: ', y_test)
                print('(' + time.asctime(time.localtime(time.time())) + ') Test acc:', test_acc_fold, 'Test time: ',test_used_time_fold )
                print('-' * 30 + 'fold  ' + str(curr_fold) + '  End: test' + '-' * 30)
                test_accuracy_allfold = np.append(test_accuracy_allfold, test_acc_fold)
                train_used_time_allfold = np.append(train_used_time_allfold, train_used_time_fold)
                test_used_time_allfold = np.append(test_used_time_allfold, test_used_time_fold)
                loss_allfold.append(history.history['loss'][-1])

                K.clear_session()

            summary = pd.DataFrame({'fold': range(1,fold+1), 'Test accuracy': test_accuracy_allfold, 'loss': loss_allfold, 'train time': train_used_time_allfold, 'test time': test_used_time_allfold})
            hyperparam = pd.DataFrame({'average acc of 10 folds': np.mean(test_accuracy_allfold), 'average loss of 10 folds': np.mean(loss_allfold), 'average train time of 10 folds': np.mean(train_used_time_allfold), 'average test time of 10 folds': np.mean(test_used_time_allfold),'epochs': args.epochs, 'lr':args.lr, 'batch size': args.batch_size},index=['dimention/sub'])
            writer = pd.ExcelWriter(args.save_dir + '/'+'summary'+ '_'+subject+'.xlsx')
            summary.to_excel(writer, 'Result', index=False)
            hyperparam.to_excel(writer, 'HyperParam', index=False)
            writer.save()
            print('10 fold average accuracy: ', np.mean(test_accuracy_allfold))
            print('10 fold average train time: ', np.mean(train_used_time_allfold))
            print('10 fold average test time: ', np.mean(test_used_time_allfold))
