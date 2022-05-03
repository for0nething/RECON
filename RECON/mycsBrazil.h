#ifndef UNTITLED2_MYCSBrazil_H
#define UNTITLED2_MYCSBrazil_H


#include "cnpy.h"
#include "type.h"

#include "util.h"
#include "data.h"
#include <cstring>
#include <fstream>
#include <random>
#include <unordered_map>


namespace Brazil{
    using std::chrono::system_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    std::random_device rd;
    std::mt19937 mt(rd());

    dtype taxiW = 1. / 14, t5W = 2. / 14, t11W = 7. / 14, t16W = 2. / 14, t20W = 2. / 14;


    cnpy::NpyArray taxiArr;
    cnpy::NpyArray t5Arr;
    cnpy::NpyArray t11Arr;
    cnpy::NpyArray t16Arr;
    cnpy::NpyArray t20Arr;
    cnpy::NpyArray joinArr;

    dtype *dp;


    dtype *taxi, *t5, *t11, *t16, *t20, *join;

    idtype taxiNum, taxiDim, t5Num, t5Dim, t11Num, t11Dim, t16Num, t16Dim, t20Num, t20Dim, joinNum, joinDim;

    dtype *taxiSim, *t5Sim, *t11Sim, *t16Sim, *t20Sim;



    std::vector<idtype> movies;


    idtype *f642Weight;



    cnpy::NpyArray loadNpy(std::string fileDir);
    void
    readTaxiNpy(int cate);
    void
    mallocTaxiArray();
    void
    loadToArr(int cate);

    void
    readTaxiNpyGlobal();
    void
    mallocTaxiArrayGlobal();
    void
    loadToArrGlobal();



    void mallocTaxiSim();
    void calTaxiSim();

    void mallocTaxiSimGlobal();
    void calTaxiSimGlobal();


    void initWeight();
    void initWeightGlobal();



    void sampleOneTaxi(idtype &ID5, idtype &ID11, idtype &ID16, idtype &ID20, idtype &f642, idtype &rowID,
                       idtype &joinID);
    void sampleBatchTaxi(int sampleSize,
                         std::vector<idtype> &ID5s,
                         std::vector<idtype> &ID11s,
                         std::vector<idtype> &ID16s,
                         std::vector<idtype> &ID20s,
                         std::vector<idtype> &f642s,
                         std::vector<idtype> &rowIDs,
                         std::vector<idtype> &joinIDs);

    void starDP(idtype ID5,
                idtype ID11,
                idtype ID16,
                idtype ID20,
                idtype f642);


    void realAddOne(idtype joinID);




    dtype getBenefitTaxi(idtype ID5,
                         idtype ID11,
                         idtype ID16,
                         idtype ID20,
                         idtype f642,
                         idtype rowID,
                         idtype joinID,
                         bool change,
                         int verbose);


    std::chrono::duration<long, std::ratio<1, 1000000>> testTaxi(dtype PROP,
                                                                 dtype epsilon,
                                                                 int saveWhere,
                                                                 int verbose
    );

    std::vector<idtype> fullCS;
    std::vector<dtype> fullCSWeight;


    cnpy::NpyArray loadNpy(std::string fileDir) {

        cnpy::NpyArray arr = cnpy::npy_load(fileDir);
        return arr;
    }

    void readTaxiNpyGlobal() {
        std::stringstream dir;
        dir.str("");

        dir << DATAPATH + "taxi-formycs/train-taxi.npy";
        taxiArr = loadNpy(dir.str());
        dir.str("");

        dir << DATAPATH + "taxi-formycs/train-t5.npy";
        t5Arr = loadNpy(dir.str());
        dir.str("");

        dir << DATAPATH + "taxi-formycs/train-t16.npy";
        t16Arr = loadNpy(dir.str());
        dir.str("");

        dir << DATAPATH + "taxi-formycs/train-t20.npy";
        t20Arr = loadNpy(dir.str());
        dir.str("");
    }

    void mallocTaxiArrayGlobal() {


        taxiNum = taxiArr.shape[0];
        taxiDim = taxiArr.shape[1];
        taxi = (dtype *) malloc(taxiNum * taxiDim * sizeof(dtype));


        t5Num = t5Arr.shape[0];
        t5Dim = t5Arr.shape[1];
        t5 = (dtype *) malloc(t5Num * t5Dim * sizeof(dtype));


        t16Num = t16Arr.shape[0];
        t16Dim = t16Arr.shape[1];
        t16 = (dtype *) malloc(t16Num * t16Dim * sizeof(dtype));


        t20Num = t20Arr.shape[0];
        t20Dim = t20Arr.shape[1];
        t20 = (dtype *) malloc(t20Num * t20Dim * sizeof(dtype));
    }

    void loadToArrGlobal() {

        readTaxiNpyGlobal();
        mallocTaxiArrayGlobal();
        memcpy(taxi, taxiArr.data<dtype>(), 1LL * taxiNum * taxiDim * sizeof(dtype));
        memcpy(t5, t5Arr.data<dtype>(), 1LL * t5Num * t5Dim * sizeof(dtype));
        memcpy(t16, t16Arr.data<dtype>(), 1LL * t16Num * t16Dim * sizeof(dtype));
        memcpy(t20, t20Arr.data<dtype>(), 1LL * t20Num * t20Dim * sizeof(dtype));
    }


    void readTaxiNpy(int cate) {
        std::stringstream dir;
        dir.str("");


        dir << DATAPATH + "taxi-formycs/train-"
            << cate << "-joined.npy";
        joinArr = loadNpy(dir.str());
        dir.str("");

        dir << DATAPATH + "taxi-formycs/train-"
            << cate << "-t11.npy";
        t11Arr = loadNpy(dir.str());
        dir.str("");

    }

    void mallocTaxiArray() {


        joinNum = joinArr.shape[0];
        joinDim = joinArr.shape[1];
        join = (dtype *) malloc(joinNum * joinDim * sizeof(dtype));


        t11Num = t11Arr.shape[0];
        t11Dim = t11Arr.shape[1];
        t11 = (dtype *) malloc(t11Num * t11Dim * sizeof(dtype));
    }

    void loadToArr(int cate) {


        readTaxiNpy(cate);
        mallocTaxiArray();
        memcpy(join, joinArr.data<dtype>(), 1LL * joinNum * joinDim * sizeof(dtype));
        memcpy(t11, t11Arr.data<dtype>(), 1LL * t11Num * t11Dim * sizeof(dtype));
    }


    void mallocTaxiSimGlobal() {


        taxiSim = (dtype *) malloc(taxiNum * taxiNum * sizeof(dtype));
        t5Sim = (dtype *) malloc(t5Num * t5Num * sizeof(dtype));
        t16Sim = (dtype *) malloc(t16Num * t16Num * sizeof(dtype));
        t20Sim = (dtype *) malloc(t20Num * t20Num * sizeof(dtype));
    }

    void calTaxiSimGlobal() {

        initSim(taxiSim, taxi, taxiNum, taxiDim, 1, taxiDim - 1);

        initSim(t5Sim, t5, t5Num, t5Dim, 2);

        initSim(t16Sim, t16, t16Num, t16Dim, 2);

        initSim(t20Sim, t20, t20Num, t20Dim, 2);
    }

    void mallocTaxiSim() {

        t11Sim = (dtype *) malloc(t11Num * t11Num * sizeof(dtype));
    }

    void calTaxiSim() {

        initSim(t11Sim, t11, t11Num, t11Dim, 2);
    }

    dtype *tp, *tp2;
    std::vector<int> f642s;

    void initWeightGlobal() {



        tp = (dtype *) malloc((taxiNum + 1) * sizeof(dtype));
        tp2 = (dtype *) malloc((taxiNum + 1) * sizeof(dtype));

        f642s.clear();
        f642Weight = (idtype *) malloc(500 * sizeof(idtype));



        for (int i = 0; i < taxiNum; i++) {
            idtype key = taxi[i * taxiDim];

            f642s.emplace_back(key);
            f642Weight[key] = 1;
            int cnt = 0;
            for (int j = 0; j < t5Num; j++) {
                int loc = j * t5Dim + 1;
                if (key == t5[loc])++cnt;
            }
            f642Weight[key] *= cnt;

            cnt = 0;
            for (int j = 0; j < t16Num; j++) {
                int loc = j * t16Dim + 1;
                if (key == t16[loc])++cnt;
            }
            f642Weight[key] *= cnt;

            cnt = 0;
            for (int j = 0; j < t20Num; j++) {
                int loc = j * t20Dim + 1;
                if (key == t20[loc])++cnt;
            }
            f642Weight[key] *= cnt;
        }
    }


    std::vector<int> joinIDs;

    void initWeight() {

        joinIDs.clear();
        joinIDs.reserve(joinNum);
        dp = (dtype *) malloc(t11Num * sizeof(dtype));
        memset(dp, 0, t11Num * sizeof(dtype));
        for (int i = 0; i < joinNum; i++)
            joinIDs.emplace_back(i);
    }

    void sampleOneTaxi(idtype &ID5,
                       idtype &ID11,
                       idtype &ID16,
                       idtype &ID20,
                       idtype &f642,
                       idtype &rowID,
                       idtype &joinID) {

        int id = joinIDs[mt() % joinIDs.size()];


        idtype idx_st = id * joinDim;

        f642 = join[idx_st];
        ID5 = join[idx_st + 1];
        ID11 = join[idx_st + 2];
        ID16 = join[idx_st + 3];
        ID20 = join[idx_st + 4];

        rowID = join[idx_st + joinDim - 1];
        joinID = id;
    }

    void sampleBatchTaxi(int sampleSize,
                         std::vector<idtype> &ID5s,
                         std::vector<idtype> &ID11s,
                         std::vector<idtype> &ID16s,
                         std::vector<idtype> &ID20s,
                         std::vector<idtype> &f642s,
                         std::vector<idtype> &rowIDs,
                         std::vector<idtype> &joinIDs) {
        ID5s.resize(sampleSize);
        ID11s.resize(sampleSize);
        ID16s.resize(sampleSize);
        ID20s.resize(sampleSize);
        f642s.resize(sampleSize);
        rowIDs.resize(sampleSize);
        joinIDs.resize(sampleSize);

        for (int i = 0; i < sampleSize; i++)
            sampleOneTaxi(ID5s[i],
                          ID11s[i],
                          ID16s[i],
                          ID20s[i],
                          f642s[i],
                          rowIDs[i],
                          joinIDs[i]);
    }

    void realAddOne(idtype joinID) {

        for (int i = 0; i < joinIDs.size(); i++) {
            if (joinIDs[i] == joinID) {
                std::swap(joinIDs[joinIDs.size() - 1], joinIDs[i]);
                joinIDs.pop_back();
                break;
            }
        }
    }


    void starDP(idtype ID5,
                idtype ID11,
                idtype ID16,
                idtype ID20,
                idtype f642) {

        memset(tp2, 0x3f, sizeof(tp2) * taxiNum);
        idtype simloc = ID5 * t5Num;
        for (int i = 0; i < t5Num; i++) {
            idtype this_f642 = t5[i * t5Dim + 1];
            dtype this_sim = t5Sim[simloc + i];
            tp2[this_f642] = std::min(tp2[this_f642], this_sim);
        }
        for (int i = 0; i < taxiNum; i++)
            tp[i] += tp2[i] * t5W;



        simloc = ID16 * t16Num;
        memset(tp2, 0x3f, sizeof(tp2) * taxiNum);
        for (int i = 0; i < t16Num; i++) {
            idtype this_f642 = t16[i * t16Dim + 1];
            dtype this_sim = t16Sim[simloc + i];
            tp2[this_f642] = std::min(tp2[this_f642], this_sim);
        }
        for (int i = 0; i < taxiNum; i++)
            tp[i] += tp2[i] * t16W;


        simloc = ID20 * t20Num;
        memset(tp2, 0x3f, sizeof(tp2) * taxiNum);
        for (int i = 0; i < t20Num; i++) {
            idtype this_f642 = t20[i * t20Dim + 1];
            dtype this_sim = t20Sim[simloc + i];
            tp2[this_f642] = std::min(tp2[this_f642], this_sim);
        }
        for (int i = 0; i < taxiNum; i++)
            tp[i] += tp2[i] * t20W;



        simloc = f642 * taxiNum;
        memset(tp2, 0x3f, sizeof(tp2) * taxiNum);

        for (int i = 0; i < taxiNum; i++) {
            idtype this_f642 = taxi[i * taxiDim];
            dtype this_sim = taxiSim[simloc + i];
            tp2[this_f642] = std::min(tp2[this_f642], this_sim);
        }
        for (int i = 0; i < taxiNum; i++)
            tp[i] += tp2[i] * taxiW;

    }

    dtype getBenefitTaxi(idtype ID5,
                         idtype ID11,
                         idtype ID16,
                         idtype ID20,
                         idtype f642,
                         idtype rowID,
                         idtype joinID,
                         bool change = false,
                         int verbose = 1) {
        memset(tp, 0, sizeof(dtype) * taxiNum);
        starDP(ID5, ID11, ID16, ID20, f642);

        dtype simSum = 0;
        dtype thisWeight = 0.;


        idtype sim_loc = ID11 * t11Num;

        for (int i = 0; i < t11Num; i++) {
            idtype tmp_f642 = t11[i * t11Dim + 1];
            dtype tmp_dp = tp[tmp_f642];
            dtype t11_sim = t11Sim[sim_loc + i];
            tmp_dp += t11_sim * t11W;
            if (tmp_dp > dp[i] && change) {
                dp[i] = tmp_dp;
                if (cs.nn[i] != -1) {
                    cs.weight[cs.nn[i]] -= f642Weight[tmp_f642];
                }
                cs.nn[i] = cs.weight.size();
                thisWeight += f642Weight[tmp_f642];
            }
            simSum += std::max(tmp_dp, dp[i]) * f642Weight[tmp_f642];
        }

        if (change) {
            cs.curSum = simSum;
            cs.curSum = cs.norm * std::log(1. + cs.f_norm * cs.curSum);

            cs.add(rowID);
            cs.weight.emplace_back(thisWeight);
            if (verbose)
                printf("    add this weight is %.2f         Current progress 【%.2f %%】\n", thisWeight,
                       100. * cs.weight.size() / cs.siz);
            realAddOne(joinID);
        }

        return cs.norm * std::log(1. + cs.f_norm * simSum) - cs.curSum;
    }


    std::chrono::duration<long, std::ratio<1, 1000000>> testBrazil(dtype PROP,
                                                                 dtype epsilon = 0.01,
                                                                 int saveWhere = 0,
                                                                 int verbose = 1
    ) {
        fullCS.clear();
        fullCSWeight.clear();

        std::chrono::duration<long, std::ratio<1, 1000000>> sim_time(0);


        loadToArrGlobal();
        mallocTaxiSimGlobal();
        calTaxiSimGlobal();
        initWeightGlobal();

        std::vector<idtype> ID5s;
        std::vector<idtype> ID11s;
        std::vector<idtype> ID16s;
        std::vector<idtype> ID20s;
        std::vector<idtype> f642s;
        std::vector<idtype> rowIDs;
        std::vector<idtype> samplejoinIDs;


        for (int cate = 0; cate <= 93; cate++) {
            auto st = system_clock::now();
            if (verbose)std::cout << "#############       Current category is " << cate << "     ##########\n";


            loadToArr(cate);
            initWeight();

            mallocTaxiSim();
            calTaxiSim();

            if (verbose)std::cout << "join N is " << joinNum << "\n";
            if (verbose)std::cout << "PROP is " << PROP << "\n";

            idtype csSize = (idtype) (PROP * joinNum);
            if (verbose)std::cout << "This cate should have [" << csSize << "]\n";


            idtype sampleEachStep = 1. / PROP * std::log(1. / epsilon) + 0.5;


            idtype ano = 1. / PROP * std::log(1. / epsilon) + 0.5;
            if (ano < sampleEachStep)
                sampleEachStep = ano;

            cs.init(t11Num, csSize);
            cs.f_norm = 1. / joinNum;

            auto en = system_clock::now();
            auto duration = duration_cast<microseconds>(en - st);
            sim_time += duration;

            while (csSize--) {
                dtype curMaxBenefit = -1;
                idtype curMaxBenefitID = 0;

                sampleBatchTaxi(sampleEachStep, ID5s, ID11s, ID16s, ID20s, f642s, rowIDs, samplejoinIDs);
                std::vector<dtype> benefit_vec(sampleEachStep);

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < sampleEachStep; i++)
                    benefit_vec[i] = getBenefitTaxi(ID5s[i], ID11s[i], ID16s[i], ID20s[i], f642s[i], rowIDs[i],
                                                    samplejoinIDs[i], 0, 0);
                idtype i = 0;
                for (auto val : benefit_vec) {
                    if (val > curMaxBenefit) {
                        curMaxBenefit = val;
                        curMaxBenefitID = i;
                    }
                    ++i;
                }
                i = curMaxBenefitID;

                if (verbose)std::cout << "Benefit is " << curMaxBenefit<<"\n";
                getBenefitTaxi(ID5s[i], ID11s[i], ID16s[i], ID20s[i], f642s[i], rowIDs[i], samplejoinIDs[i], true, 0);
            }

            fullCS.insert(fullCS.end(), cs.coresetAll.begin(), cs.coresetAll.end());
            fullCSWeight.insert(fullCSWeight.end(), cs.weight.begin(), cs.weight.end());


            if (verbose)std::cout << "Finished!\n";
        }

        if (verbose)printf("Total coreset size 【%d】\n", fullCS.size());

        if (verbose)
            std::cout << "@### 【Similarity】 Spent "
                      << double(sim_time.count()) * microseconds::period::num / microseconds::period::den
                      << " seconds.\n";

        assert(!saveWhere);
        if (!saveWhere) {
            std::stringstream dir;
            dir.str("");
            dir<<CSPATH <<"Brazilnew";
            dir<< "-"<<PROP<<"-ours.npz";
            std::cout<<"Save to "<< dir.str() <<"\n";
            cnpy::npz_save(dir.str(), "order", &fullCS[0], {fullCS.size()}, "w");
            cnpy::npz_save(dir.str(), "weight", &fullCSWeight[0],
                           {fullCSWeight.size()},
                           "a");
            dtype order_time = 0.;
            cnpy::npz_save(dir.str(), "order_time", &order_time, {1}, "a");
            printf("%s\n", cur_time());
            printf("Save finished\n");
        }
        return sim_time;
    }
}
#endif
