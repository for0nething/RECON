#ifndef UNTITLED2_DATA_H
#define UNTITLED2_DATA_H
#include "type.h"
#include <sstream>
#include "type.h"
#include "global.h"
void loadData(char * data= nullptr){
    if(!data) {
        assert(0);
    }
    std::stringstream ss;
    std::string dataName(data);
    ss.str("");
    ss << DATAPATH << dataName << "-train-X.npy";
    cnpy::NpyArray trainX = cnpy::npy_load(ss.str());
    n = trainX.shape[0];
    d = trainX.shape[1];
    X = (dtype *) malloc(n * d * sizeof(dtype));
    memcpy(X, trainX.data<dtype>(), n * d * sizeof(dtype));

    ss.str("");
    ss << DATAPATH << dataName << "-train-y.npy";
    cnpy::NpyArray trainY = cnpy::npy_load(ss.str());
    assert(trainY.shape[0] == trainX.shape[0]);
    std::cout<<"word size is "<<trainY.word_size<<"\n";
    n = trainY.shape[0];
    Y = (labeltype *) malloc(1LL * n * sizeof(labeltype));
    memcpy(Y, trainY.data<labeltype>(), n * sizeof(labeltype));

}

#endif //UNTITLED2_DATA_H


