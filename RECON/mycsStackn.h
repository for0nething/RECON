#ifndef UNTITLED2_MYCSStackn_H
#define UNTITLED2_MYCSStackn_H


#include "cnpy.h"
#include "type.h"
#include "util.h"
#include "data.h"
#include "coreset.h"
#include <cstring>
#include <fstream>
#include <random>
#include <unordered_map>


namespace stackn {
    using std::chrono::system_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    std::random_device rd;
    std::mt19937 mt(rd());


    dtype userW = 1. / 14, questionW = 2. / 14, answerW = 7. / 14;

    cnpy::NpyArray userArr;
    cnpy::NpyArray questionArr;
    cnpy::NpyArray answerArr;
    cnpy::NpyArray joinArr;

    dtype *dp;
    dtype *user, *question, *answer, *join;
    idtype userNum, userDim, questionNum, questionDim, answerNum, answerDim, joinNum, joinDim;
    dtype *userSim, *questionSim, *answerSim;

    std::vector<idtype> users;
    std::vector<idtype> fullCS;
    std::vector<dtype> fullCSWeight;    


    void freeStackn() {
        free(user);
        free(answer);
        free(question);
        free(join);
        free(userSim);
        free(answerSim);
        free(questionSim);
        free(dp);
    }

    cnpy::NpyArray loadNpy(std::string fileDir) {
        cnpy::NpyArray arr = cnpy::npy_load(fileDir);
        return arr;
    }


    void readStacknNpy(int cate) {
        std::stringstream dir;
        dir.str("");

        dir << DATAPATH + "stackn-formycs/train-"
            << cate << "-user.npy";
        userArr = loadNpy(dir.str());
        dir.str("");

        dir << DATAPATH + "stackn-formycs/train-"
            << cate << "-answer.npy";
        answerArr = loadNpy(dir.str());
        dir.str("");


        dir << DATAPATH + "stackn-formycs/train-"
            << cate << "-question.npy";
        questionArr = loadNpy(dir.str());
        dir.str("");


        dir << DATAPATH + "stackn-formycs/train-"
            << cate << "-joined.npy";
        joinArr = loadNpy(dir.str());
        dir.str("");
    }

    void mallocStacknArray() {


        userNum = userArr.shape[0];
        userDim = joinArr.shape[1];
        user = (dtype *) malloc(userNum * userDim * sizeof(dtype));


        answerNum = answerArr.shape[0];
        answerDim = answerArr.shape[1];
        answer = (dtype *) malloc(answerNum * answerDim * sizeof(dtype));


        questionNum = questionArr.shape[0];
        questionDim = questionArr.shape[1];
        question = (dtype *) malloc(questionNum * questionDim * sizeof(dtype));


        joinNum = joinArr.shape[0];
        joinDim = joinArr.shape[1];
        join = (dtype *) malloc(joinNum * joinDim * sizeof(dtype));
    }

    void loadToArr(int cate) {


        readStacknNpy(cate);
        mallocStacknArray();

        memcpy(user, userArr.data<dtype>(), 1LL * userNum * userDim * sizeof(dtype));
        memcpy(answer, answerArr.data<dtype>(), 1LL * answerNum * answerDim * sizeof(dtype));
        memcpy(question, questionArr.data<dtype>(), 1LL * questionNum * questionDim * sizeof(dtype));
        memcpy(join, joinArr.data<dtype>(), 1LL * joinNum * joinDim * sizeof(dtype));



        for (int i = 0; i < joinNum; i++)
            join[i * joinDim + 5] = i;
    }

    void mallocStacknSim() {


        userSim = (dtype *) malloc(userNum * userNum * sizeof(dtype));
        answerSim = (dtype *) malloc(answerNum * answerNum * sizeof(dtype));
        questionSim = (dtype *) malloc(questionNum * questionNum * sizeof(dtype));
    }

    void calStacknSim() {

        initSim(userSim, user, userNum, userDim, 1);

        initSim(answerSim, answer, answerNum, answerDim, 3);

        initSim(questionSim, question, questionNum, questionDim, 2);


    }

    std::vector<int> joinIDs;

    void initWeight() {

        joinIDs.clear();
        joinIDs.reserve(answerNum);
        dp = (dtype *) malloc(answerNum * sizeof(dtype));
        memset(dp, 0, answerNum * sizeof(dtype));
        for (int i = 0; i < answerNum; i++)
            joinIDs.emplace_back(i);
    }


    void sampleOneStackn(idtype & uID,
                         idtype & ID,
                         idtype & qID,
                         idtype & rowID,
                         idtype & samplejoinID) {

        int id = joinIDs[mt() % joinIDs.size()];



        idtype idx_st = id * joinDim;

        uID = join[idx_st];
        qID = join[idx_st + 1];
        ID = join[idx_st + 2];
        rowID = join[idx_st + 3];
        samplejoinID = join[idx_st + 5];
    }


    void sampleBatchStackn(int sampleSize,
                           std::vector<idtype> &uIDs,
                           std::vector<idtype> &IDs,
                           std::vector<idtype> &qIDs,
                           std::vector<idtype> &rowIDs,
                           std::vector<idtype> &joinIDs) {
        uIDs.resize(sampleSize);
        IDs.resize(sampleSize);
        qIDs.resize(sampleSize);
        rowIDs.resize(sampleSize);
        joinIDs.resize(sampleSize);

        for (int i = 0; i < sampleSize; i++)
            sampleOneStackn(uIDs[i],
                            IDs[i],
                            qIDs[i],
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


    dtype getBenefitStackn(idtype uID,
                           idtype ID,
                           idtype qID,
                           idtype rowID,
                           idtype joinID,
                           bool change = false,
                           int verbose = 1) {


        dtype simSum = 0;
        dtype thisWeight = 0.;

        idtype sim_loc_user = uID * userNum;
        idtype sim_loc_answer = ID * answerNum;
        idtype sim_loc_question = qID * questionNum;



        idtype idx_loc = 0;
        for (int i = 0; i < answerNum; i++, idx_loc += answerDim) {
            idtype _id = answer[idx_loc];
            idtype _uID = answer[idx_loc + 1];
            idtype _qID = answer[idx_loc + 2];

            dtype tempDP = answerSim[sim_loc_answer + _id] * answerW;
            tempDP += questionSim[sim_loc_question + _qID] * questionW;
            tempDP += userSim[sim_loc_user + _uID] * userW;


            if (tempDP > dp[i] && change) {
                dp[i] = tempDP;
                if (cs.nn[i] != -1) {
                    cs.weight[cs.nn[i]] -= 1;
                }
                cs.nn[i] = cs.weight.size();
                thisWeight += 1;
            }
            simSum += std::max(tempDP, dp[i]);
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


    std::chrono::duration<long, std::ratio<1, 1000000>> testStackn(dtype PROP,
                                                                   dtype epsilon = 0.01,
                                                                   int saveWhere = 0,
                                                                   int verbose = 1
    ) {
        fullCS.clear();
        fullCSWeight.clear();

        std::chrono::duration<long, std::ratio<1, 1000000>> sim_time(0);


        std::vector<idtype> uIDs;
        std::vector<idtype> IDs;
        std::vector<idtype> qIDs;
        std::vector<idtype> rowIDs;
        std::vector<idtype> samplejoinIDs;


        for (int cate = 0; cate <= 18305; cate++) {
            auto st = system_clock::now();
            if (verbose)
                std::cout << "#############       Current category is " << cate << "     ##########\n";


            loadToArr(cate);
            initWeight();

            mallocStacknSim();
            calStacknSim();


            assert(joinNum == answerNum);

            if (verbose)std::cout << "join N is " << joinNum << "\n";
            if (verbose)std::cout << "PROP is " << PROP << "\n";

            idtype csSize = (idtype) (PROP * joinNum + 0.5);
            if (verbose)std::cout << "This cate should have [" << csSize << "]\n";


            idtype sampleEachStep = 1. / PROP * std::log(1. / epsilon) + 0.5;


            idtype ano = 1. / PROP * std::log(1. / epsilon) + 0.5;
            if (ano < sampleEachStep)
                sampleEachStep = ano;

            cs.init(joinNum, csSize);
            cs.f_norm = 1. / joinNum;

            auto en = system_clock::now();
            auto duration = duration_cast<microseconds>(en - st);
            sim_time += duration;

            while (csSize--) {
                dtype curMaxBenefit = -1;
                idtype curMaxBenefitID = 0;


                std::vector<idtype> uIDs;
                std::vector<idtype> IDs;
                std::vector<idtype> qIDs;
                std::vector<idtype> rowIDs;
                std::vector<idtype> samplejoinIDs;

                sampleBatchStackn(sampleEachStep, uIDs, IDs, qIDs, rowIDs, samplejoinIDs);
                std::vector<dtype> benefit_vec(sampleEachStep);

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < sampleEachStep; i++)
                    benefit_vec[i] = getBenefitStackn(uIDs[i], IDs[i], qIDs[i], rowIDs[i], samplejoinIDs[i], 0, 0);
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
                benefit_vec[i] = getBenefitStackn(uIDs[i], IDs[i], qIDs[i], rowIDs[i], samplejoinIDs[i], 1, 0);
            }

            freeAll();
            freeStackn();

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
            dir<<CSPATH <<"stackn";
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


