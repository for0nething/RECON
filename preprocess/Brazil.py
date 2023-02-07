import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os

X_train = []
X_test = []
y_train = []
y_test = []

import datetime
def parseDatetime(s):
    pre, suf = s.split(' ')
    
    year_s, mon_s, day_s = pre.split('-')
    hour_s, minute_s, second_s = suf.split(':')
    return datetime.datetime(int(year_s), int(mon_s), int(day_s), int(hour_s), int(minute_s), int(second_s))

def timeDelta(arrLike, col1, col2):
    purchase = parseDatetime(arrLike[col1])
    approve = parseDatetime(arrLike[col2])
    delta = approve - purchase
    return delta.total_seconds()

DIR = '/home/jiayi/disk/C-craig/dataset/Brazil/'

file = 'olist_order_reviews_dataset'
review = pd.read_csv(DIR + file + '.csv')

file = 'olist_orders_dataset.csv'
order = pd.read_csv(DIR + file)

file = 'olist_order_items_dataset.csv'
orderItem = pd.read_csv(DIR + file)

file = 'olist_products_dataset.csv'
product = pd.read_csv(DIR + file)

review = review[['review_id', 'order_id', 'review_score','review_creation_date', 'review_answer_timestamp']].copy()
order = order[['order_id',  'order_status', 'order_purchase_timestamp',
              'order_approved_at', 'order_delivered_carrier_date',
              'order_delivered_customer_date', 'order_estimated_delivery_date']].copy()
orderItem = orderItem[['order_id', 'product_id',
                      'price', 'freight_value']].copy()
product = product[['product_id', 'product_photos_qty']].copy()

review.review_score = review.review_score - 1

tmp = pd.merge(review, order)
tmp = pd.merge(tmp, orderItem)
tmp = pd.merge(tmp, product)

print(tmp.shape)
tmp.dropna(inplace=True)
print(tmp.shape)

tmp['approve'] = tmp.apply(timeDelta, axis=1, args=('order_purchase_timestamp',
                                                    'order_approved_at'))
tmp['approve'] /= tmp['approve'].max()
tmp['deliver'] = tmp.apply(timeDelta, axis=1, args=('order_approved_at',
                                                    'order_delivered_carrier_date'))
tmp['deliver'] /= tmp['deliver'].max()
tmp['arrive'] = tmp.apply(timeDelta, axis=1, args=('order_delivered_carrier_date',
                                                   'order_delivered_customer_date'))
tmp['arrive'] /= tmp['arrive'].max()
tmp['review'] = tmp.apply(timeDelta, axis=1, args=('review_creation_date',
                                                   'review_answer_timestamp'))
tmp['review'] /= tmp['review'].max()

tmp['faster'] = tmp.apply(timeDelta, axis=1, args=('order_delivered_customer_date',
                                                   'order_estimated_delivery_date'))
tmp['faster'] /= tmp['faster'].max()
isDelivered_idx = tmp[tmp['order_status'] == 'delivered'].index
isCanceled_idx = tmp[tmp['order_status'] == 'canceled'].index
tmp.loc[isDelivered_idx, 'order_status'] = 0
tmp.loc[isCanceled_idx,  'order_status'] = 1
col_list = [
    'review_score',
    'order_status',
    'approve',
    'deliver',
    'arrive',
    'faster',
    'review'
]

tmp.drop([
          'review_creation_date','review_answer_timestamp',
          'order_purchase_timestamp', 'order_approved_at',
          'order_delivered_carrier_date', 'order_delivered_customer_date',
          'order_estimated_delivery_date',
         ], axis=1, inplace=True)

print(tmp.columns)
print(tmp.shape)

for col in tmp.columns:
    if col not in [
    'review_score',
    'order_status',
    'approve', 
    'deliver', 
    'arrive',
    'faster',
    'review',
    'product_id','review_id', 'order_id',
    ]:
        print(col)
        tmp[col] = (tmp[col] - tmp[col].min()) / (tmp[col].max() - tmp[col].min())

tmp.drop_duplicates(keep='first',inplace=True)
print(tmp.shape)

print(tmp.columns)

review = tmp[['review_id', 'order_id', 'review_score','review']].copy()

order = tmp[['order_id',  'order_status', 'approve', 'deliver','arrive',  'faster']].copy()

orderItem = tmp[['order_id', 'product_id',
                      'price', 'freight_value']].copy()

product = tmp[['product_id', 'product_photos_qty']].copy()

review.drop_duplicates(['order_id'],keep='first', inplace=True)
review.drop_duplicates(['review_id'],keep='first', inplace=True)

order.drop_duplicates(keep='first', inplace=True)
orderItem.drop_duplicates(keep='first', inplace=True)
product.drop_duplicates(keep='first', inplace=True)

from sklearn.utils import shuffle
rng=np.random.RandomState(123)
review = shuffle(review, random_state=rng)

print("All base data shape is ")
print(review.shape)

TrainProp = 0.5
ValProp = 0.25
TrainEnd = int(TrainProp * review.shape[0])
ValEnd = TrainEnd + int(ValProp * review.shape[0])

print(TrainEnd)
print(ValEnd)

trainReview = review[:TrainEnd].copy()
valReview = review[TrainEnd:ValEnd].copy()
testReview = review[ValEnd:].copy()

trainSet = pd.merge(trainReview, order)
trainSet = pd.merge(trainSet, orderItem)
trainSet = pd.merge(trainSet, product)

valSet = pd.merge(valReview, order)
valSet = pd.merge(valSet, orderItem)
valSet = pd.merge(valSet, product)

testSet = pd.merge(testReview, order)
testSet = pd.merge(testSet, orderItem)
testSet = pd.merge(testSet, product)

DIR = "/home/jiayi/disk/C-craig/dataset/"
dataName = "Brazilnew"

trainSet.to_csv(DIR + "{}-train.csv".format(dataName), index=False)
valSet.to_csv(DIR + "{}-val.csv".format(dataName), index=False)
testSet.to_csv(DIR + "{}-test.csv".format(dataName), index=False)

y_train = trainSet.review_score.values
y_val = valSet.review_score.values
y_test = testSet.review_score.values

trainSet.drop(['review_id', 'order_id','review_score','product_id' ], axis=1, inplace=True)
valSet.drop(['review_id', 'order_id', 'review_score', 'product_id' ], axis=1, inplace=True)
testSet.drop(['review_id', 'order_id','review_score', 'product_id' ], axis=1, inplace=True)

X_train = np.ascontiguousarray(trainSet.astype(np.float64))
X_val = np.ascontiguousarray(valSet.astype(np.float64))
X_test = np.ascontiguousarray(testSet.astype(np.float64))

print(trainSet.shape)
print(trainSet.columns)

DIR = "/home/jiayi/disk/C-craig/dataset/"
dataName = "Brazilnew"
np.save(DIR + "{}-train-X.npy".format(dataName), X_train)
np.save(DIR + "{}-test-X.npy".format(dataName), X_test)
np.save(DIR + "{}-val-X.npy".format(dataName), X_val)

np.save(DIR + "{}-train-y.npy".format(dataName), y_train)
np.save(DIR + "{}-test-y.npy".format(dataName), y_test)
np.save(DIR + "{}-val-y.npy".format(dataName), y_val)

DIR = "/home/jiayi/disk/C-craig/dataset/"
dataName = "Brazilnew"

tmp = pd.read_csv(DIR + "{}-train.csv".format(dataName))
tmp['rowID'] = np.arange(tmp.shape[0])

dataName = "Brazilnew"
mycsDIR = '/home/jiayi/disk/C-craig/dataset/{}-formycs/'.format(dataName)

tmp = tmp[['review_id', 'order_id', 'product_id', 'rowID', 
           'review_score', 'review', 'order_status',
           'approve', 'deliver', 'arrive', 'faster', 'price',
           'freight_value', 'product_photos_qty']].copy()

for cate in range(5):
    train = tmp[tmp.review_score==cate].copy()

    le = preprocessing.LabelEncoder()
    le.fit(train.review_id)
    train.review_id = le.transform(train.review_id)

    le = preprocessing.LabelEncoder()
    le.fit(train.order_id)
    train.order_id = le.transform(train.order_id)

    le = preprocessing.LabelEncoder()
    le.fit(train.product_id)
    train.product_id = le.transform(train.product_id)
    
    
    train.to_csv(mycsDIR + "train-cate-{}-joined.csv".format(cate), index=False)
    tmp_ = np.ascontiguousarray(train.values.astype(np.float64))
    np.save(mycsDIR + "train-cate-{}-joined.npy".format(cate), tmp_)
    

    review = train[['review_id', 'order_id', 'review_score','review']].copy()
    review.sort_values(by='review_id')

    order = train[['order_id',  'order_status', 'approve', 'deliver','arrive',  'faster']].copy()
    order.sort_values(by='order_id')
    
    orderItem = train[['order_id', 'rowID','product_id',
                          'price', 'freight_value']].copy()
    orderItem.sort_values(by='order_id')
    
    product = train[['product_id', 'product_photos_qty']].copy()
    
    
    review.drop_duplicates(keep='first', inplace=True)
    order.drop_duplicates(keep='first', inplace=True)
    orderItem.drop_duplicates(keep='first', inplace=True)
    product.drop_duplicates(keep='first', inplace=True)

    
    
    np.save(mycsDIR + 'train-cate-{}-review.npy'.format(cate), np.ascontiguousarray(review.values.astype(np.float64)))
    np.save(mycsDIR + 'train-cate-{}-order.npy'.format(cate), np.ascontiguousarray(order.values.astype(np.float64)))
    np.save(mycsDIR + 'train-cate-{}-orderItem.npy'.format(cate), np.ascontiguousarray(orderItem.values.astype(np.float64)))
    np.save(mycsDIR + 'train-cate-{}-product.npy'.format(cate), np.ascontiguousarray(product.values.astype(np.float64)))

    
    review.to_csv(mycsDIR + 'train-cate-{}-review.csv'.format(cate),index=False)
    order.to_csv(mycsDIR + 'train-cate-{}-order.csv'.format(cate), index=False)
    orderItem.to_csv(mycsDIR + 'train-cate-{}-orderItem.csv'.format(cate), index=False)
    product.to_csv(mycsDIR + 'train-cate-{}-product.csv'.format(cate), index=False)

