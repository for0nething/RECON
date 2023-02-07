#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
from sklearn import preprocessing

import datetime
def parseDatetime(s):
#     print('s is ',s)
    pre, suf = s.split(' ')
    
    year_s, mon_s, day_s = pre.split('-')
    hour_s, minute_s, second_s = suf.split(':')
#     retuabsrn datetime.datetime(int(year_s), int(mon_s), int(day_s), int(hour_s), int(minute_s), int(second_s))
    return datetime.datetime(int(year_s), int(mon_s), int(day_s), int(hour_s), int(minute_s), int(second_s)).date()

def parseYMD(arrLike, col1):
    YMD = parseDatetime(arrLike[col1])
    return str(YMD)

def timeDelta(arrLike, col1, col2):
    purchase = parseDatetime(arrLike[col1])
    approve = parseDatetime(arrLike[col2])
    delta = approve - purchase
    return delta.total_seconds()

X_train = []
X_test = []
y_train = []
y_test = []

DIR= "/home/jiayi/disk/gits/craig/datasets/taxi/data/"

taxi = pd.read_csv(DIR+"taxi.csv")

def readDF(ID):
    df = pd.read_csv(DIR+"tbl_{}.csv".format(ID))
    return df

t16 = readDF(16)
t16['f642'] = t16.apply(parseYMD, axis=1, args=['f405'])

t5 = readDF(5)
t20 = readDF(20)
#t23 = readDF(23)

#t24 = readDF(24)
t14 = readDF(14)
t11 = readDF(11)
#t6 = readDF(6)

t5.rename({'f188':'f642'},axis=1,inplace=True)
t20.rename({'f520':'f642'},axis=1,inplace=True)
# t23.rename({'f607':'f642'},axis=1,inplace=True)

# t24.rename({'f634':'f642'},axis=1,inplace=True)
t14.rename({'f373':'f642'},axis=1,inplace=True)
t11.rename({'f299':'f642'},axis=1,inplace=True)
# t6.rename({'f195':'f642'},axis=1,inplace=True)

t11 = t11[['f642', 'f294','f298', 'f300','f302', 'f306']].copy()
print(t11.shape)
print(t11.columns)

""" t5"""
# print(t5.shape)
# print(t5.columns)
# print(t5.f189.min(), t5.f189.max())

""" t20"""
# print(t20.shape)
# print(t20.columns)
# print(t20)
# print(t20.f189.min(), t5.f189.max())

""" t16"""
t16.drop(['f405'],axis=1,inplace=True)
t16 = t16[['f642','f406','f407','f408']].copy()
# print(t16.shape)
# print(t16.columns)
# print(t16)
# print(t20.f189.min(), t5.f189.max())

z = pd.merge(taxi, t11,left_on='f642', right_on='f642')
print(z.shape)

z = pd.merge(z, t5)
print(z.shape)

z = pd.merge(z, t20)
print(z.shape)

z = pd.merge(z, t16)
print(z.shape)

le = preprocessing.LabelEncoder()
le.fit(z.f642)
z.f642 = le.transform(z.f642)
# aUser.user_id = le.transform(aUser.user_id)

taxi = z[taxi.columns].copy().drop_duplicates()
print(taxi.shape)

t11 = z[t11.columns].copy().drop_duplicates()
print(t11.shape)

t5 = z[t5.columns].copy().drop_duplicates()
print(t5.shape)

t20 = z[t20.columns].copy().drop_duplicates()
print(t20.shape)

t16 = z[t16.columns].copy().drop_duplicates()
print(t16.shape)

taxi.dropna(inplace=True)
taxi = taxi[['f642','f643','target']].copy()
print(taxi.f643.min(), taxi.f643.max())
print(taxi.target.min(), taxi.target.max())

taxi.f643 = (taxi.f643 - (taxi.f643.min())) / (taxi.f643.max()-taxi.f643.min())
std = taxi.target.std()
print(std)
taxi.target/=std
print(taxi)

t11.dropna(inplace=True)
print(t11)

cols = ['f300','f302','f306']
for col in cols:
    True_idx = t11[col].map(lambda x: x==True)
    False_idx = t11[col].map(lambda x: x==False)
    t11.loc[True_idx, col] = 1
    t11.loc[False_idx, col] = 0

print(t11)
t11 = t11.join(pd.get_dummies(t11.f294))
t11.drop(['f294'], axis=1,inplace=True)
t11.f298 = (t11.f298 - t11.f298.min())/(t11.f298.max()-t11.f298.min())
t11["ID11"] = np.arange(t11.shape[0])
print(t11.shape)

t5.dropna(inplace=True)
print(t5)
print(t5.f189.min(), t5.f189.max())
t5.f189 = (t5.f189 - t5.f189.min())/(t5.f189.max()-t5.f189.min())

t5["ID5"] = np.arange(t5.shape[0])

t20.dropna(inplace=True)
print(t20)
print(t20.f521.min(), t20.f521.max())
print(t20.f522.min(), t20.f522.max())
print(t20.f523.min(), t20.f523.max())
t20.f521 = (t20.f521 - t20.f521.min())/(t20.f521.max()-t20.f521.min())
t20.f522 = (t20.f522 - t20.f522.min())/(t20.f522.max()-t20.f522.min())
t20.f523 = (t20.f523 - t20.f523.min())/(t20.f523.max()-t20.f523.min())

t20["ID20"] = np.arange(t20.shape[0])

t16.dropna(inplace=True)
t16.f406 = (t16.f406 - t16.f406.min())/(t16.f406.max()-t16.f406.min())
t16.f407 = (t16.f407 - t16.f407.min())/(t16.f407.max()-t16.f407.min())
t16.f408 = (t16.f408 - t16.f408.min())/(t16.f408.max()-t16.f408.min())
t16["ID16"] = np.arange(t16.shape[0])

print(taxi.shape)

rng = np.random.RandomState(123)
from sklearn.utils import shuffle
taxi = shuffle(taxi, random_state=rng)

TrainProp = 0.5
ValProp = 0.25
TrainEnd = int(TrainProp * taxi.shape[0])
ValEnd = TrainEnd + int(ValProp * taxi.shape[0])

trainTaxi = taxi[:TrainEnd]
valTaxi = taxi[TrainEnd:ValEnd]
testTaxi = taxi[ValEnd:]

print(trainTaxi.shape)
print(valTaxi.shape)
print(testTaxi.shape)

print(trainTaxi.columns)

DATASET_DIR = '/home/jiayi/disk/C-craig/dataset/'
dataset = 'taxi-single'
y_train = trainTaxi.target
y_val = valTaxi.target
y_test = testTaxi.target

X_train = np.ascontiguousarray(trainTaxi[['f643']].copy().values.astype(np.float64))
X_val = np.ascontiguousarray(valTaxi[['f643']].copy().values.astype(np.float64))
X_test = np.ascontiguousarray(testTaxi[['f643']].copy().values.astype(np.float64))

np.save( DATASET_DIR + "{}-train-X.npy".format(dataset),X_train)
np.save(DATASET_DIR + "{}-val-X.npy".format(dataset),X_val)
np.save( DATASET_DIR + "{}-test-X.npy".format(dataset),X_test)

np.save(DATASET_DIR + "{}-train-y.npy".format(dataset),y_train)
np.save(DATASET_DIR + "{}-val-y.npy".format(dataset),y_val)
np.save(DATASET_DIR + "{}-test-y.npy".format(dataset),y_test)

trainSet = pd.merge(trainTaxi, t11)
trainSet = pd.merge(trainSet, t5)
trainSet = pd.merge(trainSet, t20)
trainSet = pd.merge(trainSet, t16)
print(trainSet.shape)

valSet = pd.merge(valTaxi, t11)
valSet = pd.merge(valSet, t5)
valSet = pd.merge(valSet, t20)
valSet = pd.merge(valSet, t16)
print(valSet.shape)

testSet = pd.merge(testTaxi, t11)
testSet = pd.merge(testSet, t5)
testSet = pd.merge(testSet, t20)
testSet = pd.merge(testSet, t16)
print(testSet.shape)

z = 0.9
print(z **20)

z = 0.8
print(z ** 20)

z = 0.7
print(z ** 20)

DATASET_DIR = '/home/jiayi/disk/C-craig/dataset/'
dataset = 'taxi'

y_train = trainSet.target.copy()
y_val = valSet.target.copy()
y_test = testSet.target.copy()

trainSet.to_csv(DATASET_DIR + "{}-train.csv".format(dataset), index=False)
valSet.to_csv(DATASET_DIR + "{}-val.csv".format(dataset), index=False)
testSet.to_csv(DATASET_DIR + "{}-test.csv".format(dataset), index=False)

trainSet.drop(['target','f642',"ID11","ID5","ID20","ID16"],axis=1,inplace=True)
valSet.drop( ['target','f642',"ID11","ID5","ID20","ID16"],axis=1,inplace=True)
testSet.drop(['target','f642',"ID11","ID5","ID20","ID16"],axis=1,inplace=True)

X_train = np.ascontiguousarray(trainSet.values.astype(np.float64))
X_val = np.ascontiguousarray(valSet.values.astype(np.float64))
X_test = np.ascontiguousarray(testSet.values.astype(np.float64))

np.save( DATASET_DIR + "{}-train-X.npy".format(dataset),X_train)
np.save(DATASET_DIR + "{}-val-X.npy".format(dataset),X_val)
np.save( DATASET_DIR + "{}-test-X.npy".format(dataset),X_test)

np.save(DATASET_DIR + "{}-train-y.npy".format(dataset),y_train)
np.save(DATASET_DIR + "{}-val-y.npy".format(dataset),y_val)
np.save(DATASET_DIR + "{}-test-y.npy".format(dataset),y_test)

DATASET_DIR = '/home/jiayi/disk/C-craig/dataset/'
dataset = 'taxi'

df = pd.read_csv(DATASET_DIR + "{}-train-X.csv".format(dataset))
print(df.shape)

print(df.columns)

print(df.target.value_counts())

df['rowID'] = np.arange(df.shape[0])

le = preprocessing.LabelEncoder()
le.fit(df.target)
df.target = le.transform(df.target)

print(len(df.target.unique()))
cate_list = df.target.unique()

le = preprocessing.LabelEncoder()
le.fit(df.ID5)
df.ID5 = le.transform(df.ID5)

le = preprocessing.LabelEncoder()
le.fit(df.ID11)
df.ID11 = le.transform(df.ID11)

le = preprocessing.LabelEncoder()
le.fit(df.ID16)
df.ID16 = le.transform(df.ID16)

le = preprocessing.LabelEncoder()
le.fit(df.ID20)
df.ID20 = le.transform(df.ID20)

le = preprocessing.LabelEncoder()
le.fit(df.f642)
df.f642 = le.transform(df.f642)

taxi = df[taxi.columns].copy().drop_duplicates()
print(taxi.shape)

t11 = df[t11.columns].copy().drop_duplicates()
print(t11.shape)

t5 = df[t5.columns].copy().drop_duplicates()
print(t5.shape)

t20 = df[t20.columns].copy().drop_duplicates()
print(t20.shape)

t16 = df[t16.columns].copy().drop_duplicates()
print(t16.shape)

taxi.sort_values("f642",inplace=True)
t5.sort_values("ID5",inplace=True)
t20.sort_values("ID20",inplace=True)
t16.sort_values("ID16",inplace=True)
t11.sort_values("ID11",inplace=True)

print(taxi.columns)
t5 = t5[['ID5',    'f642', 'f189']].copy()
t20 = t20[['ID20', 'f642', 'f521', 'f522', 'f523']].copy()
t16 = t16[['ID16', 'f642', 'f406', 'f407', 'f408']].copy()
t11 = t11[['ID11', 'f642', 'f298', 'f300', 'f302', 'f306', 'Booted in Error',
       'Duplicate Case', 'Executed', 'NJS Released', 'Other',
       'Paid in the Field', 'Redeemed', 'Reduced', 'Salvage History',
       'Salvage and Total Loss', 'Salvage/Total Loss/Export', 'Sold',
       'Sold Abandoned', 'Stolen Vehicle', 'Total Loss', 'Towed in Error',
       'Vehicle Not Towed', 'Zero Released']].copy()
print(t5.columns)
print(t20.columns)
print(t16.columns)
print(t11.columns)

dataset = 'taxi'
DATASET_DIR = '/home/jiayi/disk/C-craig/dataset/{}-formycs/'.format(dataset)

taxi.to_csv(DATASET_DIR + "train-taxi.csv", index=False)
t5.to_csv(DATASET_DIR + "train-t5.csv", index=False)
t20.to_csv(DATASET_DIR + "train-t20.csv", index=False)
t16.to_csv(DATASET_DIR + "train-t16.csv", index=False)

taxi_ = np.ascontiguousarray(taxi.values.astype(np.float64))
t5_ = np.ascontiguousarray(t5.values.astype(np.float64))
t20_ = np.ascontiguousarray(t20.values.astype(np.float64))
t16_ = np.ascontiguousarray(t16.values.astype(np.float64))

np.save(DATASET_DIR + "train-taxi.npy", taxi_)
np.save(DATASET_DIR + "train-t5.npy", t5_)
np.save(DATASET_DIR + "train-t20.npy", t20_)
np.save(DATASET_DIR + "train-t16.npy", t16_)

uni = np.sort(df.target.unique())
print(uni)
# print(df.target.unique())

print(df.columns)

df = df[['f642', 'ID5', 'ID11', 'ID16', 'ID20', 'f643', 'target', 'f298', 'f300', 'f302', 'f306',
       'Booted in Error', 'Duplicate Case', 'Executed', 'NJS Released',
       'Other', 'Paid in the Field', 'Redeemed', 'Reduced', 'Salvage History',
       'Salvage and Total Loss', 'Salvage/Total Loss/Export', 'Sold',
       'Sold Abandoned', 'Stolen Vehicle', 'Total Loss', 'Towed in Error',
       'Vehicle Not Towed', 'Zero Released', 'f189', 'ID5', 'f521',
       'f522', 'f523', 'f406', 'f407', 'f408', 'rowID']].copy()

for cate in uni:
    tmpDF = df[df['target'] == cate].copy()
    
    le = preprocessing.LabelEncoder()
    le.fit(tmpDF.ID11)
    tmpDF.ID11 = le.transform(tmpDF.ID11)
    
#     print(tmpDF.shape)
    tmpDF.to_csv(DATASET_DIR + "train-{}-joined.csv".format(cate), index=False)
    tmp_ = np.ascontiguousarray(tmpDF.values.astype(np.float64))
    np.save(DATASET_DIR + "train-{}-joined.npy".format(cate), tmp_)
    
    tmpt11 = tmpDF[t11.columns].copy()
    tmpt11.drop_duplicates(inplace=True) 
    tmpt11.to_csv(DATASET_DIR + "train-{}-t11.csv".format(cate), index=False)
    tmpt11_ = np.ascontiguousarray(tmpt11.values.astype(np.float64))
    np.save(DATASET_DIR + "train-{}-t11.npy".format(cate), tmpt11_)
    print(tmpt11_.shape)

print(len(t11.f642.unique()))
print(len(t5.f642.unique()))
print(len(t20.f642.unique()))
print(len(t16.f642.unique()))
# print(len(t11.f642.unique()))

