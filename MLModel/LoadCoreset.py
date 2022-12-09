import numpy as np
from MLModel.Global import *

def LoadCoreset(coreset_from, data, subset_size, batch=0, sampleSize=0):
    assert coreset_from == 'diskOurs'
    if coreset_from == 'diskOurs':
        assert batch==0
        if batch==0:
            if subset_size == 0.00001:
                file_name = CSPATH+"inuse/{}-0.00001-ours.npz".format(data)
            else:
                file_name = CSPATH+'inuse/{}-{}-ours.npz'.format(data, str(subset_size))
    print("【Load file path】 is ", file_name)


    if file_name != '':
        print(f'reading from {file_name}')
        dataset = np.load(f'{file_name}')
        order, weights, total_ordering_time = dataset['order'], dataset['weight'], dataset['order_time']
        print(" 【Coreset size】 is ", order.shape)
        return order, weights, total_ordering_time