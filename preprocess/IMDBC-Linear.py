#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
from sklearn import preprocessing

X_train = []
X_test = []
y_train = []
y_test = []

from scipy import sparse
def transMultihot(df, rowName, colName, IDName, onehotName='s'):
    tmp = df[colName].factorize()
    df.drop(colName,axis=1, inplace=True)
    df.insert(df.shape[1],colName,tmp[0])
    
    values = np.ones(df.shape[0])
    rows = df[rowName].values
    cols = df[colName].values
    
    sparse_matrix = sparse.coo_matrix((values, (rows,cols)))
    ar = sparse_matrix.toarray()
    sm = ar.sum(axis=1)  
    
    idxs = sm>0 
    IDs = np.arange(ar.shape[0])
    
    IDs = IDs[idxs]
    ARs = ar[idxs]
    
    col_name_list = ['{}{}'.format(onehotName, i) for i in range(ARs.shape[1])]
    col_name_list = [IDName] + col_name_list
       
    assert IDs.shape[0] == ARs.shape[0]
    IDs = IDs.reshape(-1,1)

    z = np.concatenate((IDs,ARs),axis=1)

    
    ret = pd.DataFrame(z, columns=col_name_list)
    ret[IDName] = ret[IDName].astype(np.int64)
    return ret

get_ipython().run_cell_magic('time', '', "\nDIR = '/home/jiayi/disk/neurocard/datasets/job/'\n\nfile = 'title.csv'\ntitle = pd.read_csv(DIR+file)\n\nfile = 'info_type.csv'\nit = pd.read_csv(DIR+file)\n\nfile = 'movie_info.csv'\nmi = pd.read_csv(DIR+file)\n\nfile = 'movie_info_idx.csv'\nmix = pd.read_csv(DIR+file)\n\nfile = 'name.csv'\nname = pd.read_csv(DIR+file)\n\nfile = 'cast_info.csv'\nci = pd.read_csv(DIR+file)\n\nfile = 'movie_companies.csv'\nmc = pd.read_csv(DIR+file)\n\nfile = 'company_name.csv'\ncn = pd.read_csv(DIR+file)")

def changeToFloor(arrLike, col):

    colValue = arrLike[col]
    colValue = np.around(arrLike[col],1)
    
    return colValue

def LoadIMDBC(Large=0,dataName="", saveCSV=False, useFor='test'):
    global X_train, X_test, X_val, y_val, y_train, y_test

    z = mix.copy()

    votes = z[z['info_type_id']==100].copy()
    rating = z[z['info_type_id']==101].copy()

    votes['info'] = votes['info'].astype(int)
    useVotes = votes[votes['info']>100].copy()

    useVotes.rename(columns={'info':'votes'},inplace=True)
    useVotes = useVotes[['movie_id', 'votes']]

    MAX = useVotes.votes.max()
    MIN = useVotes.votes.min()
    useVotes.votes = (useVotes.votes - MIN)/(MAX - MIN)

    rating['info'] = rating['info'].astype(np.double)
    useRating = rating.copy()

    useRating.rename(columns={'info':'rating'},inplace=True)

    useRating = useRating[['movie_id', 'rating']]
    useRating['rating'] = useRating['rating'].astype(np.double)

    
    
    
    
    

    useRating['rating'] = useRating.apply(changeToFloor, axis=1, args=['rating'])

    if useFor == 'train': 
        midLE = preprocessing.LabelEncoder()
        midLE.fit(useRating.rating)
        useRating['rating'] = midLE.transform(useRating.rating)

        useRating['rating']=useRating['rating'].astype(int)
        useRating.rating -=1

    useMIX = pd.merge(useVotes, useRating)
    print(useMIX.shape)
    print(useMIX.columns)

    useMI = mi.copy()
    color = useMI[useMI['info_type_id']==2].copy()
    genres = useMI[useMI['info_type_id']==3].copy()

    color.rename(columns={'info':'color'},inplace=True)

    color = color[['movie_id', 'color']]

    BWIndex = color[color['color']=='Black and White'].index
    ColorIndex = color[color['color']=='Color'].index
    color.loc[BWIndex,'color'] = 0
    color.loc[ColorIndex,'color'] = 1

    genres.rename(columns={'info':'genres'},inplace=True)
    genres = genres[['movie_id', 'genres']]
    genres.drop_duplicates(inplace=True)

    genres = transMultihot(genres, 'movie_id', 'genres', IDName='movie_id', onehotName='s')

    useMI = pd.merge(color, genres)

    print(useMI.shape)
    print(useMI.columns)

    if Large==0: 
        useCI = ci[ci['role_id']==4].copy()
    else:
        useCI= ci.copy()

    useCI = useCI[['person_id', 'movie_id']]
    print(useCI.shape)
    print(useCI.columns)
    

    useNAME = name.copy()

    mIndex = useNAME[useNAME['gender']=='m'].index
    fIndex = useNAME[useNAME['gender']=='f'].index

    useNAME.loc[mIndex,'gender'] = 1
    useNAME.loc[fIndex,'gender'] = 0

    genderNA = ~useNAME['gender'].isna()
    # purchaseNA = ~tmp['order_purchase_timestamp'].isna()

    useNAME = useNAME[genderNA]

    useNAME.rename(columns={'id':'person_id'},inplace=True)
    useNAME = useNAME[['person_id', 'gender']]
    print(useNAME.shape)
    print(useNAME.columns)

    useTITLE = title.copy()
    useTITLE.rename(columns={'id':'movie_id'},inplace=True)
    yearNA = ~useTITLE.production_year.isna()
    kindNA = ~useTITLE.kind_id.isna()
    yearNA = yearNA & kindNA
    useTITLE = useTITLE[yearNA]

    useTITLE = useTITLE[['movie_id', 'production_year','kind_id']].copy()
    MIN = useTITLE.production_year.min()
    MAX = useTITLE.production_year.max()

    useTITLE['production_year'] = (useTITLE['production_year'] - MIN)/(MAX - MIN)
    useTITLE = useTITLE.join(pd.get_dummies(useTITLE.kind_id))
    useTITLE.rename(columns={1:'k1',2:'k2',3:'k3',4:'k4',6:'k6',7:'k7'},inplace=True)

    useTITLE.drop(['kind_id'],axis=1, inplace=True)
    print(useTITLE.shape)
    print(useTITLE.columns)
    

    
    useMC = mc.copy()
    useCN = cn.copy()
    useCN.rename(columns={'id':'company_id'},inplace=True)
    le = preprocessing.LabelEncoder()
    le.fit(useCN.country_code)
    useCN['country_code'] = le.transform(useCN.country_code)
    
    joinedMC = pd.merge(useMC, useCN)
    
    tMC = joinedMC[['company_id', 'country_code','movie_id']].copy()
    

    MAX = tMC.country_code.max()
    tMC['country_code'] = (tMC['country_code']/MAX)
    

    useTITLE.drop_duplicates(useTITLE.columns,inplace=True)
    useMIX.drop_duplicates(subset=['movie_id'], keep='first', inplace=True)
    useMIX.drop_duplicates(useMIX.columns,inplace=True)
    useCI.drop_duplicates(useCI.columns,inplace=True)
    useNAME.drop_duplicates(useNAME.columns,inplace=True)
    useMI.drop_duplicates(subset=['movie_id'], keep='first', inplace=True)
    useMI.drop_duplicates(useMI.columns,inplace=True)
    
    
    

    useTITLE.drop_duplicates(inplace=True)
    useCI.drop_duplicates(inplace=True)
    useNAME.drop_duplicates(inplace=True)
    useMI.drop_duplicates(inplace=True)
    useMIX.drop_duplicates(inplace=True)
    tMC.drop_duplicates(inplace=True)
    
    
    
    

    z = pd.merge(useTITLE, useMIX)
    print(z.shape)
    z = pd.merge(z, useCI)
    print(z.shape)
    z = pd.merge(z, useNAME)
    print(z.shape)
    z = pd.merge(z, useMI)
    print(z.shape)
    

    print(z.columns)
    print(z.shape)

    z = pd.merge(z, tMC)
    print(z.columns)
    print(z.shape)
    
    
    from sklearn.utils import shuffle
    z = shuffle(z, random_state=123)
    

    movieIDs= title.id.unique()
    movieIDs = shuffle(movieIDs, random_state=123)
    
    
    trainSize = int(0.5 * movieIDs.shape[0])
    valSize = trainSize + int(0.25 * movieIDs.shape[0])
    
    
    trainMovies = movieIDs[:trainSize]
    trainTMP = pd.DataFrame(trainMovies.reshape(-1,1), columns=["movie_id"])
    trainData = pd.merge(z, trainTMP)
    
    valMovies = movieIDs[trainSize:valSize]
    valTMP = pd.DataFrame(valMovies.reshape(-1,1), columns=["movie_id"])
    valData = pd.merge(z, valTMP)
    
    testMovies = movieIDs[valSize:]
    testTMP = pd.DataFrame(testMovies.reshape(-1,1), columns=["movie_id"])
    testData = pd.merge(z, testTMP)
    

    
    
    
    

    y_train = trainData.rating.values
    y_val = valData.rating.values
    y_test = testData.rating.values
    
    

    if saveCSV:
        trainData.to_csv('/home/jiayi/disk/C-craig/dataset/{}-train.csv'.format(dataName), index=False)
        valData.to_csv('/home/jiayi/disk/C-craig/dataset/{}-val.csv'.format(dataName), index=False)
        testData.to_csv('/home/jiayi/disk/C-craig/dataset/{}-test.csv'.format(dataName), index=False)
    
    
    trainData.drop(['rating'], axis=1, inplace=True)
    valData.drop(['rating'], axis=1, inplace=True)
    testData.drop(['rating'], axis=1, inplace=True)
    
    trainData.drop(['person_id','movie_id', 'company_id'], axis=1, inplace=True)
    valData.drop(['person_id', 'movie_id', 'company_id'], axis=1, inplace=True)
    testData.drop(['person_id', 'movie_id', 'company_id'], axis=1, inplace=True)
     
        
    print("Train Data shape   ", trainData.shape)
    print("Test Data shape   ", testData.shape)
    print("Val Data shape   ", valData.shape)
    
    print(trainData.columns)
    X_train = np.ascontiguousarray(trainData.values.astype(np.float64))
    X_val = np.ascontiguousarray(valData.values.astype(np.float64))
    X_test = np.ascontiguousarray(testData.values.astype(np.float64))
    
    print(X_train.shape)
    print(y_train.shape)
    
    return z

dataNameList = ["IMDBCLinear","IMDBCLinearC++" ]
parameterList = [0,0]

# dataNameList = ["IMDBLargeCLinear","IMDBLargeCLinearC++" ]
# parameterList = [1,1]

useForList = ["test", "train"]

for dataName, param,useFor in zip(dataNameList, parameterList, useForList):
    LoadIMDBC(param, dataName, saveCSV=True,useFor=useFor) 
    np.save("/home/jiayi/disk/C-craig/dataset/{}-train-X.npy".format(dataName),X_train)
    np.save("/home/jiayi/disk/C-craig/dataset/{}-train-y.npy".format(dataName),y_train)

    np.save("/home/jiayi/disk/C-craig/dataset/{}-val-X.npy".format(dataName),X_val)
    np.save("/home/jiayi/disk/C-craig/dataset/{}-val-y.npy".format(dataName),y_val)

    np.save("/home/jiayi/disk/C-craig/dataset/{}-test-X.npy".format(dataName),X_test)
    np.save("/home/jiayi/disk/C-craig/dataset/{}-test-y.npy".format(dataName),y_test)

# dataName = "IMDBCLinearC++"
# dataName = "IMDBCLinear"
dataName = "IMDBLargeCLinearC++"

df = pd.read_csv('/home/jiayi/disk/C-craig/dataset/{}-train.csv'.format(dataName))

# dataName = "IMDBCLinearC++"
dataName2 = "IMDBLargeCLinear"

df2 = pd.read_csv('/home/jiayi/disk/C-craig/dataset/{}-train.csv'.format(dataName2))

print(df.iloc[:3])
print(df2.iloc[:3])

print(df.columns)
print(df.shape)

print(np.unique(df.rating))

midLE = preprocessing.LabelEncoder()
midLE.fit(df.movie_id)
df['movie_id'] = midLE.transform(df.movie_id)

pidLE = preprocessing.LabelEncoder()
pidLE.fit(df.person_id)
df['person_id'] = pidLE.transform(df.person_id)

cidLE = preprocessing.LabelEncoder()
cidLE.fit(df.company_id)
df['company_id'] = cidLE.transform(df.company_id)

PROP = 1
trainData = df.values
print(trainData.shape)
print(trainData[:5,:])
np.save('/home/jiayi/disk/C-craig/dataset/{}-joined-prop-{}.npy'.format(dataName, PROP), np.ascontiguousarray(trainData.astype(np.float64)))

print(midLE.classes_.shape)
print(pidLE.classes_.shape)
print(cidLE.classes_.shape)
num = midLE.classes_.shape[0]
num = num * pidLE.classes_.shape[0]
num = num * cidLE.classes_.shape[0]
print(num)
assert num< 4* (10**18)
print("\n【 Passed 】")

uni = np.unique(df[['movie_id', 'person_id']], axis=0)
print(uni.shape)
uni = np.unique(df[['movie_id', 'person_id']], axis=0)
print(uni.shape) 

print(df.shape)

uni = df[['movie_id', 'person_id', 'company_id']].copy()
# uni = np.unique(df[['movie_id', 'person_id', 'company_id']], axis=0)
print(uni.shape)
print(uni)
rowNumMap = np.zeros((uni.shape[0],2), np.int64)
i = 0
# for row in uni.values:
#     print(row)
for row in uni.values:
#     if row.sha
#     print(row)
#     print(row.shape)
    x,y,z = row
    x = np.int64(x)
    y = np.int64(y)
    z = np.int64(z)
    
    rowNumMap[i,0] = (x+1) + (y+1)*(10**5) + (z+1) * (10**11)
    assert 0 <= rowNumMap[i,0] and rowNumMap[i,0] < (4*(10**18))
#     print(rowNumMap[i,0])
    rowNumMap[i,1] = i
    i = i + 1

mycsDIR = '/home/jiayi/disk/C-craig/dataset/{}-formycs/'.format(dataName)
np.save(mycsDIR + 'idMap.npy', np.ascontiguousarray(rowNumMap))

print(rowNumMap)

CATE = len(np.unique(df.rating))
print("Cate num is ",CATE)

Databackup = df.copy()

for cate in range(CATE + 1):
    print("#"*10 ,' '*5, '【cate】 ', cate, ' '*10, '#'*10)
    trainData = Databackup[Databackup['rating'] == cate]
    

    mixColumns = ['movie_id', 'votes', 'rating']
    mixNotUniqued = trainData[mixColumns].copy()
    mixUniqued = mixNotUniqued.drop_duplicates(mixNotUniqued.columns).copy()
    mixUniqued.sort_values(['movie_id'], inplace=True)
    print('【Movie_info_idx】')
    print(mixUniqued.shape)
    print(len(np.unique(mixUniqued.movie_id)))
    
    

    miColumns = ['movie_id', 'color', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
       's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
       's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27',
       's28', 's29']
    miNotUniqued = trainData[miColumns].copy()
    miUniqued = miNotUniqued.drop_duplicates(miNotUniqued.columns).copy()
    miUniqued.sort_values(['movie_id'], inplace=True)
    print('【Movie_info】')
    print(miUniqued.shape)
    print(len(np.unique(miUniqued.movie_id)))
    
    

    ciColumns = ['person_id', 'movie_id']
    ciNotUniqued = trainData[ciColumns].copy()
    ciUniqued = ciNotUniqued.drop_duplicates(ciNotUniqued.columns).copy()
    print('【Cast_info】')
    print(ciUniqued.shape)
    print('in cast_info movie_id unique ', len(np.unique(ciUniqued.movie_id)))
    print('in cast_info person_id unique ', len(np.unique(ciUniqued.person_id)))
    
    

    nameColumns = ['person_id', 'gender']
    nameNotUniqued = trainData[nameColumns].copy()
    nameUniqued = nameNotUniqued.drop_duplicates(nameNotUniqued.columns).copy()
    nameUniqued.sort_values(['person_id'], inplace=True)
    print('【Name】')
    print(nameUniqued.shape)
    print(len(np.unique(nameUniqued.person_id)))

    titleColumns = ['movie_id', 'production_year', 'k1', 'k2', 'k3', 'k4', 'k6', 'k7']
    titleNotUniqued = trainData[titleColumns].copy()
    titleUniqued = titleNotUniqued.drop_duplicates(titleNotUniqued.columns).copy()
    titleUniqued.sort_values(['movie_id'], inplace=True)
    print('【Title】')
    print(titleUniqued.shape)
    print(len(np.unique(titleUniqued.movie_id)))
    
    

    mcColumns = ['movie_id', 'company_id', 'country_code']
    mcNotUniqued = trainData[mcColumns].copy()
    mcUniqued = mcNotUniqued.drop_duplicates(mcNotUniqued.columns).copy()
    mcUniqued.sort_values(['movie_id'], inplace=True)
    print('【Movie Company】')
    print(mcUniqued.shape)
    print(len(np.unique(mcUniqued.movie_id)))
    
    
    mycsDIR = '/home/jiayi/disk/C-craig/dataset/{}-formycs/'.format(dataName)
    np.save(mycsDIR + 'train-cate-{}-mix.npy'.format(cate), np.ascontiguousarray(mixUniqued.values.astype(np.float64)))
    np.save(mycsDIR + 'train-cate-{}-mi.npy'.format(cate), np.ascontiguousarray(miUniqued.values.astype(np.float64)))
    np.save(mycsDIR + 'train-cate-{}-ci.npy'.format(cate), np.ascontiguousarray(ciUniqued.values.astype(np.float64)))
    np.save(mycsDIR + 'train-cate-{}-name.npy'.format(cate), np.ascontiguousarray(nameUniqued.values.astype(np.float64)))
    np.save(mycsDIR + 'train-cate-{}-title.npy'.format(cate), np.ascontiguousarray(titleUniqued.values.astype(np.float64)))
    np.save(mycsDIR + 'train-cate-{}-mc.npy'.format(cate), np.ascontiguousarray(mcUniqued.values.astype(np.float64)))

    

    mixUniqued.to_csv(mycsDIR + 'train-cate-{}-mix.csv'.format(cate),index=False)
    miUniqued.to_csv(mycsDIR + 'train-cate-{}-mi.csv'.format(cate), index=False)
    ciUniqued.to_csv(mycsDIR + 'train-cate-{}-ci.csv'.format(cate), index=False)
    nameUniqued.to_csv(mycsDIR + 'train-cate-{}-name.csv'.format(cate), index=False)
    titleUniqued.to_csv(mycsDIR + 'train-cate-{}-title.csv'.format(cate), index=False)
    mcUniqued.to_csv(mycsDIR + 'train-cate-{}-mc.csv'.format(cate), index=False)
    
    
    

print(np.unique(df.rating))

z = np.unique(df.rating)
print(z[85])
# print(len(np.unique()))

