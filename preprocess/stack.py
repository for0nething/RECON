import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
from sklearn import preprocessing

DIR = '/home/jiayi/disk/stackData/'

user = pd.read_csv(DIR + 'user' + '.csv',usecols=['id','site_id', 'reputation', 'upvotes', 'downvotes'] )
question = pd.read_csv(DIR + 'question' + '.csv', usecols=['id', 'site_id', 'score','view_count', 'favorite_count'])
answer = pd.read_csv(DIR + 'answer' + '.csv', usecols=['id', 'site_id', 'question_id','owner_user_id','score'])

useUser= user.copy()
useAnswer = answer.copy()
useQuestion = question.copy()

useUser.rename(columns={'id':'user_id'},inplace=True)
useAnswer.rename(columns={'owner_user_id':'user_id', 'score':'Y'},inplace=True)
useQuestion.rename(columns={'id':'question_id'},inplace=True)

z = pd.merge(useAnswer, useUser)
z = pd.merge(z, useQuestion)
inUser = z[['user_id', 'site_id', 'reputation', 'upvotes', 'downvotes']].copy()
inAnswer = z[['id', 'site_id', 'question_id', 'Y', 'user_id']].copy()
inQuestion = z[['question_id', 'site_id', 'score', 'view_count', 'favorite_count']].copy()
inUser.drop_duplicates(inplace=True)
inAnswer.drop_duplicates(inplace=True)
inQuestion.drop_duplicates(inplace=True)

print(inUser.shape)
print(inAnswer.shape)
print(inQuestion.shape)

print(inUser.site_id.min(), inUser.site_id.max())
print(inUser.user_id.min(), inUser.user_id.max())
inUser['Uid'] = 1000 * inUser.user_id + inUser.site_id
inAnswer['Uid'] = 1000 * inAnswer.user_id + inAnswer.site_id

inQuestion['Qid'] = 1000 * inQuestion.question_id + inQuestion.site_id
inAnswer['Qid'] = 1000 * inAnswer.question_id + inAnswer.site_id

le = preprocessing.LabelEncoder()
le.fit(inAnswer.Uid)
inAnswer.Uid = le.transform(inAnswer.Uid)
inUser.Uid = le.transform(inUser.Uid)
print(inAnswer.Uid.min(), inAnswer.Uid.max())
print(inUser.Uid.min(), inUser.Uid.max())

le = preprocessing.LabelEncoder()
le.fit(inAnswer.Qid)
inAnswer.Qid = le.transform(inAnswer.Qid)
inQuestion.Qid = le.transform(inQuestion.Qid)
print(inAnswer.Qid.min(), inAnswer.Qid.max())
print(inQuestion.Qid.min(), inQuestion.Qid.max())

print(inUser.iloc[:3,:])
tu = inUser[['reputation', 'upvotes' ,'downvotes']].copy()
tu.drop_duplicates(inplace=True)
print(tu.shape)

tu = inUser[['reputation', 'upvotes' ,'downvotes','site_id']].copy()
tu.drop_duplicates(inplace=True)
print(tu.shape)

tu['newUid'] = np.arange(tu.shape[0])
newU = pd.merge(inUser, tu) 
print(newU.shape)
print(newU.columns)
print(newU.iloc[:3,:])

print(inQuestion.iloc[:3,:])
tq = inQuestion[['score', 'view_count']].copy()
tq.drop_duplicates(inplace=True)
print(tq.shape)

tq['newQid'] = np.arange(tq.shape[0])
newQ = pd.merge(inQuestion, tq) 
print(newQ.shape)
print(newQ.columns)
print(newQ.iloc[:3,:])

z = pd.merge(inAnswer, newQ)
z = pd.merge(z, newU)
print(z.columns)
print(z.shape)

doAnswer = z[['id','newUid','newQid','Y']].copy().drop_duplicates()
doQuestion = z[['newQid','score', 'view_count']].copy().drop_duplicates()
doUser = z[['newUid','site_id', 'reputation','upvotes','downvotes']].copy().drop_duplicates()

print(doAnswer.shape)
print(doQuestion.shape)
print(doUser.shape)

doJoin = pd.merge(doAnswer, doQuestion)
doJoin = pd.merge(doJoin, doUser)

STD = doJoin.reputation.std()

doUserBackup = doUser.copy() 

print(doUser.iloc[:3,:])
doUser = doUser.join(pd.get_dummies(doUser.site_id, prefix='st'))
doUser.upvotes = (doUser.upvotes - doUser.upvotes.min()) / (doUser.upvotes.max() - doUser.upvotes.min())
doUser.downvotes = (doUser.downvotes - doUser.downvotes.min()) / (doUser.downvotes.max() - doUser.downvotes.min())

# doUser.reputation /= doUser.reputation.std()
# doUser.reputation /= STD

print(doUser.iloc[:3,:])

doAnswerBackup = doAnswer.copy()
doAnswer.Y = (doAnswer.Y - doAnswer.Y.min()) / (doAnswer.Y.max() - doAnswer.Y.min())

print(doAnswer.columns)
print(doAnswer.iloc[:3,:])

doQuestionBackup = doQuestion.copy()

print(doQuestion.columns)
print(doQuestion.iloc[:3,:])

doQuestion.score = (doQuestion.score - doQuestion.score.min()) / (doQuestion.score.max() - doQuestion.score.min())
doQuestion.view_count = (doQuestion.view_count - doQuestion.view_count.min()) / (doQuestion.view_count.max() - doQuestion.view_count.min())

print(doQuestion.iloc[:3,:])

doingUser = doUser.copy()

rng = np.random.RandomState(123)
from sklearn.utils import shuffle
doingUser = shuffle(doingUser, random_state=rng)

TrainProp = 0.5
ValProp = 0.25
TrainEnd = int(TrainProp * doingUser.shape[0])
ValEnd = TrainEnd + int(ValProp * doingUser.shape[0])

print(doingUser.columns)

doingUser.reputation /= doingUser.reputation.std()

trainUser = doingUser[:TrainEnd].copy()
valUser = doingUser[TrainEnd:ValEnd].copy()
testUser = doingUser[ValEnd:].copy()

print(doingUser.columns)
DATASET_DIR = '/home/jiayi/disk/C-craig/dataset/'
dataset = 'stackn-single'
y_train = trainUser.reputation
y_val = valUser.reputation
y_test = testUser.reputation

trainUser.drop(['newUid', 'site_id', 'reputation'], axis=1, inplace=True)
valUser.drop(['newUid', 'site_id', 'reputation'], axis=1, inplace=True)
testUser.drop(['newUid', 'site_id', 'reputation'], axis=1, inplace=True)

X_train = np.ascontiguousarray(trainUser.values.astype(np.float64))
X_val = np.ascontiguousarray(valUser.values.astype(np.float64))
X_test = np.ascontiguousarray(testUser.values.astype(np.float64))

np.save( DATASET_DIR + "{}-train-X.npy".format(dataset),X_train)
np.save(DATASET_DIR + "{}-val-X.npy".format(dataset),X_val)
np.save( DATASET_DIR + "{}-test-X.npy".format(dataset),X_test)

np.save(DATASET_DIR + "{}-train-y.npy".format(dataset),y_train)
np.save(DATASET_DIR + "{}-val-y.npy".format(dataset),y_val)
np.save(DATASET_DIR + "{}-test-y.npy".format(dataset),y_test)

DATASET_DIR = '/home/jiayi/disk/C-craig/dataset/'
dataset = 'stackn'

doingUser = doUser.copy()
doingUser.reputation/=STD

rng = np.random.RandomState(123)
from sklearn.utils import shuffle
doingUser = shuffle(doingUser, random_state=rng)

TrainProp = 0.5
ValProp = 0.25
TrainEnd = int(TrainProp * doingUser.shape[0])
ValEnd = TrainEnd + int(ValProp * doingUser.shape[0])

print(doingUser.columns)

# User
trainUser = doingUser[:TrainEnd].copy()
valUser = doingUser[TrainEnd:ValEnd].copy()
testUser = doingUser[ValEnd:].copy()

# join
trainSet = pd.merge(trainUser, doAnswer)
trainSet = pd.merge(trainSet,  doQuestion)

valSet = pd.merge(valUser, doAnswer)
valSet = pd.merge(valSet,  doQuestion)

testSet = pd.merge(testUser, doAnswer)
testSet = pd.merge(testSet,  doQuestion)

y_train = trainSet.reputation
y_val = valSet.reputation
y_test = testSet.reputation

trainSet.to_csv(DATASET_DIR + "{}-train-X.csv".format(dataset), index=False)
valSet.to_csv(DATASET_DIR + "{}-val-X.csv".format(dataset), index=False)
testSet.to_csv(DATASET_DIR + "{}-test-X.csv".format(dataset), index=False)

trainSet.drop(['newUid', 'newQid', 'id', 'site_id', 'reputation'], axis=1, inplace=True)
valSet.drop(['newUid', 'newQid', 'id', 'newUid', 'site_id', 'reputation'], axis=1, inplace=True)
testSet.drop(['newUid', 'newQid', 'id', 'newUid', 'site_id', 'reputation'], axis=1, inplace=True)

print(trainSet.shape)
print(trainSet.columns)
X_train = np.ascontiguousarray(trainSet.values.astype(np.float64))
X_val = np.ascontiguousarray(valSet.values.astype(np.float64))
X_test = np.ascontiguousarray(testSet.values.astype(np.float64))

np.save( DATASET_DIR + "{}-train-X.npy".format(dataset),X_train)
np.save(DATASET_DIR + "{}-val-X.npy".format(dataset),X_val)
np.save( DATASET_DIR + "{}-test-X.npy".format(dataset),X_test)

np.save(DATASET_DIR + "{}-train-y.npy".format(dataset),y_train)
np.save(DATASET_DIR + "{}-val-y.npy".format(dataset),y_val)
np.save(DATASET_DIR + "{}-test-y.npy".format(dataset),y_test)

# doAnswer.to_csv('/home/jiayi/disk/C-craig/dataset/stackn-formycs/doAnswer.csv', index=False)
# doUser.to_csv('/home/jiayi/disk/C-craig/dataset/stackn-formycs/doUser.csv', index=False)
# doQuestion.to_csv('/home/jiayi/disk/C-craig/dataset/stackn-formycs/doQuestion.csv', index=False)

doAnswer = pd.read_csv('/home/jiayi/disk/C-craig/dataset/stackn-formycs/doAnswer.csv')
doUser = pd.read_csv('/home/jiayi/disk/C-craig/dataset/stackn-formycs/doUser.csv')
doQuestion = pd.read_csv('/home/jiayi/disk/C-craig/dataset/stackn-formycs/doQuestion.csv')

DATASET_DIR = '/home/jiayi/disk/C-craig/dataset/'
dataset = 'stackn'

dfBackup = pd.read_csv(DATASET_DIR + "{}-train-X.csv".format(dataset))
print(dfBackup.shape)

df = dfBackup.copy()
print(df.columns)

df.rename(columns={'reputation':'target'},inplace=True)
doUser.rename(columns={'reputation':'target'},inplace=True)
doUser.drop(['site_id','target'],axis=1,inplace=True)

df['rowID'] = np.arange(df.shape[0])

le = preprocessing.LabelEncoder()
le.fit(df.target)
df.target = le.transform(df.target)

DATASET_DIR = '/home/jiayi/disk/C-craig/dataset/'
dataset = 'stacknC++'

y_train = df.target
y_train = np.ascontiguousarray(y_train.values.astype(np.int64))
dfC = df.drop(['rowID','newUid', 'newQid', 'id', 'site_id', 'target'], axis=1)
X_train = np.ascontiguousarray(dfC.values.astype(np.float64))

testy = np.load(DATASET_DIR + "{}-train-y.npy".format(dataset))
print(np.unique(testy))
print(len(np.unique(testy)))
print(np.min(testy), np.max(testy))

np.save( DATASET_DIR + "{}-train-X.npy".format(dataset),X_train)
np.save(DATASET_DIR + "{}-train-y.npy".format(dataset),y_train)

print(dataset)

print(dfC.columns)

print(np.sort(df.target.unique()))
print(len(df.target.unique()))
print(df.shape)

print(np.unique(y_train))
print(len(np.unique(y_train)))
print(np.min(y_train), np.max(y_train))

print(df.target.value_counts())

uni = np.sort(df.target.unique())
print(uni.shape)
print(uni)

df = df [['newUid', 'newQid', 'id',  'rowID','site_id', 'target', 'upvotes', 'downvotes', 'st_0', 'st_1', 'st_2', 'st_3', 'st_4', 'st_5', 'st_6', 'st_7', 'st_8', 'st_9', 'st_10', 'st_11', 'st_12', 'st_13', 'st_14', 'st_15', 'st_16', 'st_17', 'st_18', 'st_19', 'st_20', 'st_21', 'st_22', 'st_23', 'st_24', 'st_25', 'st_26', 'st_27', 'st_28', 'st_29', 'st_30', 'st_31', 'st_32', 'st_33', 'st_34', 'st_35', 'st_36', 'st_37', 'st_38', 'st_39', 'st_40', 'st_41', 'st_42', 'st_43', 'st_44', 'st_45', 'st_46', 'st_47', 'st_48', 'st_49', 'st_50', 'st_51', 'st_52', 'st_53', 'st_54', 'st_55', 'st_56', 'st_57', 'st_58', 'st_59', 'st_60', 'st_61', 'st_62', 'st_63', 'st_64', 'st_65', 'st_66', 'st_67', 'st_68', 'st_69', 'st_70', 'st_71', 'st_72', 'st_73', 'st_74', 'st_75', 'st_76', 'st_77', 'st_78', 'st_79', 'st_80', 'st_81', 'st_82', 'st_83', 'st_84', 'st_85', 'st_86', 'st_87', 'st_88', 'st_89', 'st_90', 'st_91', 'st_92', 'st_93', 'st_94', 'st_95', 'st_96', 'st_97', 'st_98', 'st_99', 'st_100', 'st_101', 'st_102', 'st_103', 'st_104', 'st_105', 'st_106', 'st_107', 'st_108', 'st_109', 'st_110', 'st_111', 'st_112', 'st_113', 'st_114', 'st_115', 'st_116', 'st_117', 'st_118', 'st_119', 'st_120', 'st_121', 'st_122', 'st_123', 'st_124', 'st_125', 'st_126', 'st_127', 'st_128', 'st_129', 'st_130', 'st_131', 'st_132', 'st_133', 'st_134', 'st_135', 'st_136', 'st_137', 'st_138', 'st_139', 'st_140', 'st_141', 'st_142', 'st_143', 'st_144', 'st_145', 'st_146', 'st_147', 'st_148', 'st_149', 'st_150', 'st_151', 'st_152', 'st_153', 'st_154', 'st_155', 'st_156', 'st_157', 'st_158', 'st_159', 'st_160', 'st_161', 'st_162', 'st_163', 'st_164', 'st_165', 'st_166', 'st_167', 'st_168', 'st_169', 'st_170', 'st_171', 'st_172',  'Y', 'score', 'view_count']].copy()

dataset = 'stackn'
DATASET_DIR = '/home/jiayi/disk/C-craig/dataset/{}-formycs/'.format(dataset)

for cate in uni:
    print("#"*20, " "*10, cate, " "*10, "#"*20)
    tmpDF = df[df['target'] == cate].copy()
    

    le = preprocessing.LabelEncoder()
    le.fit(tmpDF.newUid)
    tmpDF.newUid = le.transform(tmpDF.newUid)
    

    le = preprocessing.LabelEncoder()
    le.fit(tmpDF.newQid)
    tmpDF.newQid = le.transform(tmpDF.newQid)

    le = preprocessing.LabelEncoder()
    le.fit(tmpDF.id)
    tmpDF.id = le.transform(tmpDF.id)
    
    

    tmpDF.sort_values("id",inplace=True)
    tmpDF.to_csv(DATASET_DIR + "train-{}-joined.csv".format(cate), index=False)
    tmp_ = np.ascontiguousarray(tmpDF.values.astype(np.float64))
    np.save(DATASET_DIR + "train-{}-joined.npy".format(cate), tmp_)
    
     
    tmpUser = tmpDF[doUser.columns].copy()
    tmpUser.drop_duplicates(inplace=True)
    tmpUser.sort_values("newUid",inplace=True)
    tmpUser.to_csv(DATASET_DIR + "train-{}-user.csv".format(cate), index=False)
    tmpUser = np.ascontiguousarray(tmpUser.values.astype(np.float64))
    np.save(DATASET_DIR + "train-{}-user.npy".format(cate), tmpUser)

    
    tmpQuestion = tmpDF[doQuestion.columns].copy()
    tmpQuestion.drop_duplicates(inplace=True) 
    tmpQuestion.sort_values("newQid",inplace=True)
    tmpQuestion.to_csv(DATASET_DIR + "train-{}-question.csv".format(cate), index=False)
    tmpQuestion = np.ascontiguousarray(tmpQuestion.values.astype(np.float64))
    np.save(DATASET_DIR + "train-{}-question.npy".format(cate), tmpQuestion)
    print(tmpQuestion.shape)
    
    
    tmpAnswer = tmpDF[doAnswer.columns].copy()
    tmpAnswer.drop_duplicates(inplace=True)
    tmpAnswer.sort_values("id",inplace=True)
    tmpAnswer.to_csv(DATASET_DIR + "train-{}-answer.csv".format(cate), index=False)
    tmpAnswer = np.ascontiguousarray(tmpAnswer.values.astype(np.float64))
    np.save(DATASET_DIR + "train-{}-answer.npy".format(cate), tmpAnswer)
    

