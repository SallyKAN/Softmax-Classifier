import numpy as np   # numeriacal computing
import h5py
import time
from Assignment1.LR import LogisticRegression


def normalization(input_data):
    input_data = np.array(input_data, dtype = float)
    data_num, dim1, dim2 = np.shape(input_data)
    for i in range(0, data_num):
        X = input_data[i,0:28,0:28]
        X_scaled=(X-X.min())/(X.max()-X.min())
        input_data[i, 0:28, 0:28] = X_scaled
    return input_data

def reduceDime(data):
    reducedData = []
    for i in range(data.shape[0]):
        item = data[i]
        after = item.flatten()
        reducedData.append(list(after))
    return reducedData

def cnf_matrix(test_label, pre_label):
    types = 10
    cnf = np.zeros([types, types],dtype=int)
    for i in range(0, len(pre_label)):
        cnf[test_label[i], pre_label[i]] += 1
    # plt.show()
    return cnf

if __name__ == '__main__':

    #load data
    start = time.time()
    with h5py.File('Input/images_training.h5', 'r') as H:
        data = np.copy(H['data'])
    with h5py.File('Input/labels_training.h5', 'r') as H:
        label = np.copy(H['label'])
    with h5py.File('Input/labels_testing_2000.h5', 'r') as H:
        test_label = np.copy(H['label'])
    with h5py.File('Input/images_testing.h5', 'r') as H:
        test_data = np.copy(H['data'])
    print("loding time: {} s".format(time.time()-start))
    # pre-process the input data
    start2 = time.time()

    # normalize data
    data = normalization(data)
    test_data = normalization(test_data)

    # reduce dimension of data
    train_data = np.asarray(reduceDime(data))
    train_label = np.asarray(reduceDime(label))
    test_data_2000 = np.asarray(reduceDime(test_data))[0:2000]
    test_label = np.asarray(reduceDime(test_label))
    test_data_3000 = np.asarray(reduceDime(test_data))[2000:]
    train_label = train_label.T
    train_label = train_label[0]
    Y = train_label
    train_label = (np.arange(np.max(Y) + 1) == Y[:, None]).astype(int)
    print("pre-processing time: {}".format(time.time() - start2))

    # build the training model
    model = LogisticRegression(num_iters = 300,
                alpha=0.00001,
                with_reg = True,
                with_monentum = True)
    results = model.train(train_data.T, train_label.T, Y, test_data_2000.T,test_label.T[0], print_loss=True)
    label_prediction_test = results["label_prediction_test"]

    # print the confusion matrix of test set
    # cfn = cnf_matrix(label_prediction_test,test_label)
    # print(cfn)

    # output the results
    w = results["w"]
    b = results["b"]
    y_prediction_test = model.predict(w,b,test_data_3000.T)
    with h5py.File('Output/predicted_labels.h5', 'w') as H:
        H.create_dataset('label', data=y_prediction_test)

