#ifndef UNTITLED2_UTIL_H
#define UNTITLED2_UTIL_H


#include "type.h"
#include "global.h"
#include <ctime>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

char * cur_time(){
    time_t now = time(0);
    char* dt = ctime(&now);
    dt[strlen(dt) -1 ]='\0';
    return dt;
}

void test(dtype *z){
    for(idtype i = 0; i < std::min((idtype)20, n * d); i++)
        std::cout << z[i] <<" ";
    puts("");
    return;
}

inline dtype distance(idtype idx, idtype idy){
    
    dtype ret = 0;
    idx = Map[idx];
    idy = Map[idy];

    for(idtype i = 0; i < d; i++)
        ret += (X[idx * d + i] - X[idy * d + i]) * (X[idx * d + i] - X[idy * d + i]);
    return ret;
}

inline dtype tryAdd(dtype* cur, idtype element){
    
    dtype sim_sum = 0;
    #pragma omp parallel for schedule (static) reduction(+:sim_sum)
    for(idtype i = 0; i < n; i++)
        sim_sum += std::max(cur[i], similarity[element * n + i]);

    return norm * std::log(1. + f_norm * sim_sum) - curSum;
}
inline void realAdd(dtype* cur, idtype element){
    
    curSum = 0;
    for(idtype i = 0; i < n; i++){
        if(similarity[element * n + i] > cur[i]) {
            cur[i] = similarity[element * n + i];
            if(nn[i]!=-1) {

                --weight[nn[i]];
            }
            nn[i] = element;
            ++weight[element];
        }
        curSum += cur[i];
    }
    curSum = norm * std::log(1. + f_norm * curSum);
    coreset.emplace_back(element);
    ++cSize;
    return;
}
void initSimilarity(int verbose=1){
    

    dtype max_similarity = 0;

    if(verbose)printf("Start to cal similarity %s\n",cur_time());
    for(idtype i = 0; i < n; i++)
        #pragma omp parallel for schedule (static)
        for(idtype j = i + 1; j < n; j++){
            similarity[i * n + j] = -distance(i, j);

        }
    if(verbose)printf("Finish to cal similarity %s\n",cur_time());
    for(idtype i = 0; i < n; i++)
        for(idtype j = i + 1; j < n; j++){

            max_similarity = std::max(max_similarity, -similarity[i * n + j]);
        }
    if(verbose)std::cout << "max similarity is " << max_similarity << "\n";
    #pragma omp parallel for schedule (guided)
    for(idtype i = 0; i < n; i++)
        for(idtype j = i + 1; j < n; j++) {
            similarity[i * n + j] = (max_similarity + similarity[i * n + j]) / max_similarity;
        }

    for(idtype i = 0; i < n; i++)
        #pragma omp parallel for schedule (static)
        for(idtype j = 0; j < i; j++)
            similarity[i * n + j] = similarity[j * n + i];
    for(idtype i = 0; i < n; i++)
        similarity[i * n + i] = 1;

    return;
}
void initPQ(){
    
    while(!pq.empty())pq.pop();
    for(idtype i = 1; i < n; i++)
        pq.push(std::make_pair(tryAdd(maxSim, i),i));
    return;
}


void initCategories(int verbose=1){
    
    N = n;
    cateNum.clear();



    for(idtype i = 0; i < N; i++) {

        if(Y[i]==-1)Y[i]=0;

        ++cateNum[Y[i]];
    }

    cateCnt = cateNum.size();
    std::cout<<"cateCnt is "<<cateCnt<<"\n";
    for(int i = 0 ; i < cateCnt; i++) {
        assert(cateNum.find(i) != cateNum.end());
    }
    if(verbose)printf("——————————    Cate cnt is 【%d】\n", cateCnt);
    for(int i= 0 ; i < cateCnt; i++)
        if(verbose)printf(" |||   Cate [%3d]  has  [%8d]\n", i, cateNum[i]);



    real_coreset_size = 0;
    for(int cate = 0; cate < cateCnt; cate++){
        n = cateNum[cate];
        int k = 1.0 * n / N * target_coreset_size + 0.5;
        real_coreset_size += k;
    }
    weight_vec.clear();
    coreset.clear();
    coresetAll.clear();
    coresetAll.reserve(real_coreset_size);

}

std::vector<dtype> lazyVec;
bool cmpLazyIMDB(int i, int j){
    return lazyVec[i] > lazyVec[j];
}

bool cmpLazy(int i, int j){
    return lazy[i] > lazy[j];
}


inline dtype Dist(idtype u1, idtype u2, dtype * data, idtype dim, idtype st_id = 1, idtype end_id=-1){

    dtype ret = 0.;
    if(end_id == -1)
        end_id = dim;
    idtype u1Loc = u1 * dim;
    idtype u2Loc = u2 * dim;
    for(idtype i = st_id; i < end_id; i++)
        ret += (data[u1Loc + i] - data[u2Loc + i]) * (data[u1Loc + i] - data[u2Loc + i]);
    return ret;
}

void initSim(dtype * sim, dtype * data, idtype num, idtype dim, idtype st_id=1,idtype end_id=-1){

    if(end_id==-1)
        end_id = dim;
    dtype maxDis = 0.;
    #pragma omp parallel for schedule(guided)
    for(idtype i = 0; i < num; i++) {
        idtype now = i * num;
        for (idtype j = i + 1; j < num; j++) {
            sim[now + j] = -Dist(i, j, data, dim, st_id, end_id);
        }
    }
    idtype now = 0;
    for(idtype i = 0;i < num;i++) {
        for (idtype j = i + 1; j < num; j++)
            maxDis = std::max(maxDis, -sim[now+ j]);
        now += num;
    }


    #pragma omp parallel for schedule(guided)
    for(idtype i = 0; i < num; i++) {
        idtype now = i * num;
        for (idtype j = i + 1; j < num; j++)
            sim[now + j] = (maxDis + sim[now + j]) / maxDis;
    }

    #pragma omp parallel for schedule(guided)
    for(idtype i = 0; i < n; i++)

        for(idtype j = 0; j < i; j++)
            sim[i * num + j] = sim[j * num + i];

    for(idtype i = 0; i < n; i++)
        sim[i * n + i] = 1;
    return;



}


struct CS{
    int n;
    int siz;
    dtype curSum = 0.;

    dtype norm = 1./std::log(2.);
    dtype f_norm;

    idtype* nn;
    std::vector<idtype> coresetAll;
    std::vector<dtype> weight;

    void init(int n_, int size_){
        n = n_;
        curSum = 0;
        siz = size_;
        coresetAll.clear();
        weight.clear();

        coresetAll.reserve(size_);
        weight.reserve(size_);

        nn = (idtype *)malloc(n * sizeof(idtype));
        memset(nn, -1, n * sizeof(idtype));
        f_norm = 1./(2. * n);
    }
    void add(idtype id_){
        coresetAll.emplace_back(id_);
    }
}cs;


#endif

