import numpy as np
from MLModel.Global import *
def load_dataset(dataset, prop=0.1, regression=False):
    assert dataset in ['IMDBCLinear', 'IMDBLargeCLinear', 'Brazilnew', 'IMDBC5', 'IMDBLargeC5', 'taxi', 'stackn']

    X_train = np.load(DATAPATH + "{}-train-X.npy".format(dataset))
    X_val = np.load(DATAPATH + "{}-val-X.npy".format(dataset))
    X_test = np.load(DATAPATH + "{}-test-X.npy".format(dataset))
    y_train = np.load(DATAPATH + "{}-train-y.npy".format(dataset))
    y_val = np.load(DATAPATH + "{}-val-y.npy".format(dataset))
    y_test = np.load(DATAPATH + "{}-test-y.npy".format(dataset))

    if regression == False:
        assert  dataset in ['IMDBC5','IMDBLargeC5', 'Brazilnew']
        print("Is Multi class")
        if dataset in ['IMDBC5', 'IMDBLargeC5', 'Brazilnew']:
            num_class = 5
        print("Num class  ", num_class)
        if dataset in ['Brazil5']:
            y_train-=1
            y_val-=1
            y_test-=1
        print(np.unique(y_train))
        print(np.unique(y_val))
        print(np.unique(y_test))
        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
        y_test = y_test.astype(np.int32)
        y_train = np.eye(num_class)[y_train]
        y_val = np.eye(num_class)[y_val]
        y_test = np.eye(num_class)[y_test]
    elif not regression:
        y_train = np.reshape(y_train, (-1, 1))
        y_val = np.reshape(y_val, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))
    print(f'Training size: {len(y_train)}, Test size: {len(y_test)}')
    return X_train, y_train, X_val, y_val, X_test, y_test

