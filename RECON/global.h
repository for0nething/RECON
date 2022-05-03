#ifndef UNTITLED2_GLOBAL_H
#define UNTITLED2_GLOBAL_H
#include<map>
#include<string>
#include<omp.h>
#include "cnpy.h"


const std::string DATAPATH ="/home/jiayi/disk/C-craig/dataset/";
const std::string CSPATH ="/home/jiayi/disk/C-craig/inuse/";
const int tc = 16;
dtype *X;
labeltype *Y;
dtype *similarity;
idtype n, d, N;
idtype * Map;
std::map<idtype,int> cateNum;
int cateCnt;
dtype alpha = 1.;
idtype target_coreset_size;
idtype real_coreset_size;
idtype* nn;
dtype* maxSim;
dtype* weight;
dtype * lazy;
idtype * idx;
idtype * invidx;
std::vector<dtype> weight_vec;
std::priority_queue<std::pair<dtype, idtype> > pq;
std::vector<idtype> coreset;
std::vector<idtype> coresetAll;
dtype curSum;
dtype f_norm;
dtype norm;
idtype cSize;

void freeAll(){
    free(Map);
    free(lazy);
    free(invidx);
    free(idx);
    free(similarity);
    free(nn);
    free(maxSim);
    free(weight);
}
#endif //UNTITLED2_GLOBAL_H

