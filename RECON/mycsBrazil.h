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


    cnpy::NpyArray reviewArr;
    cnpy::NpyArray orderArr;
    cnpy::NpyArray orderItemArr;
    cnpy::NpyArray productArr;
    cnpy::NpyArray joinArr;


    dtype *dp;
    dtype *review, *order, *orderItem, *product, *join;
    idtype reviewNum, reviewDim, orderNum, orderDim, orderItemNum, orderItemDim, productNum, productDim, joinNum, joinDim;
    dtype *reviewSim, *orderSim, *orderItemSim, *productSim;


    cnpy::NpyArray loadNpy(std::string fileDir);
    void readBrazilnewNpy(int cate);
    void mallocBrazilnewArray();
    void loadToArr(int cate);


    void mallocBrazilnewSim();
    void calBrazilnewSim();
    void initWeight();


    void sampleOneBrazilnew(idtype &uID, idtype &ID, idtype &qID, idtype &rowID,
                            idtype &samplejoinID);
    void sampleBatchBrazilnew(int sampleSize,
                              std::vector <idtype> &uIDs,
                              std::vector <idtype> &IDs,
                              std::vector <idtype> &qIDs,
                              std::vector <idtype> &rowIDs,
                              std::vector <idtype> &joinIDs);

    void realAddOne(idtype joinID);


    dtype getBenefitBrazilnew(idtype uID,
                              idtype ID,
                              idtype qID,
                              idtype rowID,
                              idtype joinID,
                              bool change,
                              int verbose);

    std::chrono::duration<long, std::ratio<1, 1000000>> testBrazil(dtype PROP,
                                                                      dtype epsilon,
                                                                      int saveWhere,
                                                                      int verbose
    );

    std::vector <idtype> fullCS;
    std::vector <dtype> fullCSWeight;
    dtype rW = 33. / 100, oW = 33. / 100, orderItemW = 33. / 100, pW= 1./100;


    void freeBrazilnew() {
        free(review);
        free(orderItem);
        free(order);
        free(product);
        free(join);
        free(reviewSim);
        free(orderItemSim);
        free(orderSim);
        free(productSim);
        free(dp);
    }


    cnpy::NpyArray loadNpy(std::string fileDir) {

        cnpy::NpyArray arr = cnpy::npy_load(fileDir);
        return arr;
    }


    void readBrazilnewNpy(int cate) {
        std::stringstream dir;
        dir.str("");


        dir <<  DATAPATH<< "Brazilnew-formycs/train-cate-"
            << cate << "-review.npy";
        reviewArr = loadNpy(dir.str());
        dir.str("");


        dir << DATAPATH<< "Brazilnew-formycs/train-cate-"
            << cate << "-orderItem.npy";
        orderItemArr = loadNpy(dir.str());
        dir.str("");


        dir << DATAPATH<< "Brazilnew-formycs/train-cate-"
            << cate << "-order.npy";
        orderArr = loadNpy(dir.str());
        dir.str("");


        dir << DATAPATH<< "Brazilnew-formycs/train-cate-"
            << cate << "-product.npy";
        productArr = loadNpy(dir.str());
        dir.str("");


        dir << DATAPATH<< "Brazilnew-formycs/train-cate-"
            << cate << "-joined.npy";
        joinArr = loadNpy(dir.str());
        dir.str("");
    }


    void mallocBrazilnewArray() {


        reviewNum = reviewArr.shape[0];
        reviewDim = reviewArr.shape[1];
        review = (dtype *) malloc(reviewNum * reviewDim * sizeof(dtype));


        orderItemNum = orderItemArr.shape[0];
        orderItemDim = orderItemArr.shape[1];
        orderItem = (dtype *) malloc(orderItemNum * orderItemDim * sizeof(dtype));


        orderNum = orderArr.shape[0];
        orderDim = orderArr.shape[1];
        order = (dtype *) malloc(orderNum * orderDim * sizeof(dtype));

        productNum = productArr.shape[0];
        productDim = productArr.shape[1];
        product = (dtype *) malloc(productNum * productDim * sizeof(dtype));


        joinNum = joinArr.shape[0];
        joinDim = joinArr.shape[1];
        join = (dtype *) malloc(joinNum * joinDim * sizeof(dtype));
    }

    void loadToArr(int cate) {

        readBrazilnewNpy(cate);
        mallocBrazilnewArray();

        memcpy(review, reviewArr.data<dtype>(), 1LL * reviewNum * reviewDim * sizeof(dtype));
        memcpy(orderItem, orderItemArr.data<dtype>(), 1LL * orderItemNum * orderItemDim * sizeof(dtype));
        memcpy(order, orderArr.data<dtype>(), 1LL * orderNum * orderDim * sizeof(dtype));
        memcpy(product, productArr.data<dtype>(), 1LL * productNum * productDim * sizeof(dtype));
        memcpy(join, joinArr.data<dtype>(), 1LL * joinNum * joinDim * sizeof(dtype));
    }

    void mallocBrazilnewSim() {


        reviewSim = (dtype *) malloc(reviewNum * reviewNum * sizeof(dtype));
        orderItemSim = (dtype *) malloc(orderItemNum * orderItemNum * sizeof(dtype));
        orderSim = (dtype *) malloc(orderNum * orderNum * sizeof(dtype));
        productSim = (dtype *) malloc(productNum * productNum * sizeof(dtype));
    }

    void calBrazilnewSim() {

        initSim(reviewSim, review, reviewNum, reviewDim, 3);

        initSim(orderSim, order, orderNum, orderDim, 1);

        initSim(orderItemSim, orderItem, orderItemNum, orderItemDim, 3);

        initSim(productSim, product, productNum, productDim, 1);



    }


    std::vector<int> joinIDs;
    void initWeight(){

        joinIDs.clear();
        joinIDs.reserve(orderItemNum);
        dp = (dtype *)malloc(orderItemNum * sizeof(dtype));
        memset(dp,0, orderItemNum * sizeof(dtype));
        for(int i= 0 ;i < orderItemNum; i++)
            joinIDs.emplace_back(i);
    }


    void sampleOneBrazilnew(idtype &rID,
                            idtype &oID,
                            idtype &pID,
                            idtype &rowID,
                            idtype &joinID){

        int id = joinIDs[mt()% joinIDs.size()];

        idtype idx_st = id * joinDim;
        rID    = join[idx_st];
        oID    = join[idx_st + 1];
        pID    = join[idx_st + 2];
        rowID  = join[idx_st + 3];
        joinID = id;
    }


    void sampleBatchBrazilnew(int sampleSize,
                              std::vector<idtype>& rIDs,
                              std::vector<idtype>& oIDs,
                              std::vector<idtype>& pIDs,
                              std::vector<idtype>& rowIDs,
                              std::vector<idtype>& joinIDs){
        rIDs.resize(sampleSize);
        oIDs.resize(sampleSize);
        pIDs.resize(sampleSize);
        rowIDs.resize(sampleSize);
        joinIDs.resize(sampleSize);

        for(int i = 0; i < sampleSize; i ++)
            sampleOneBrazilnew(rIDs[i],
                               oIDs[i],
                               pIDs[i],
                               rowIDs[i],
                               joinIDs[i]);
    }

    void realAddOne(idtype joinID){

        for(int i = 0 ;i < joinIDs.size();i++){
            if(joinIDs[i] == joinID){
                std::swap(joinIDs[joinIDs.size()-1 ], joinIDs[i]);
                joinIDs.pop_back();
                break;
            }
        }
    }

    dtype getBenefitBrazilnew(idtype rID,
                              idtype oID,
                              idtype pID,
                              idtype rowID,
                              idtype joinID,
                              bool change=false,
                              int verbose=1){

        dtype simSum = 0;
        dtype thisWeight = 0.;

        idtype sim_loc_review = oID * orderNum;
        idtype sim_loc_order = oID * orderNum;
        idtype sim_loc_orderItem = joinID * orderItemNum;
        idtype sim_loc_product = pID * productNum;



        idtype idx_loc = 0;
        for(int i = 0, jID=0 ; i < orderItemNum; i++, idx_loc+=orderItemDim, jID++){
            idtype oid_  = orderItem[idx_loc];
            idtype rowID_ = orderItem[idx_loc + 1];
            idtype pID_ = orderItem[idx_loc + 2];


            dtype tempDP = reviewSim[sim_loc_review + oid_] * rW;
            tempDP += orderSim[sim_loc_order + oid_] * oW;
            tempDP += orderItemSim[sim_loc_orderItem + jID] * orderItemW;
            tempDP += productSim[sim_loc_product + pID_] * pW;

            if(tempDP > dp[i] && change){
                dp[i] = tempDP;
                if(cs.nn[i] !=-1){
                    cs.weight[cs.nn[i]] -= 1;
                }
                cs.nn[i] = cs.weight.size();
                thisWeight += 1;
            }
            simSum += std::max(tempDP, dp[i]);
        }


        if(change) {
            cs.curSum = simSum;
            cs.curSum = cs.norm * std::log(1. + cs.f_norm * cs.curSum);

            cs.add(rowID);
            cs.weight.emplace_back(thisWeight);
            if(verbose)
                printf("    add this weight is %.2f         Current progress 【%.2f %%】\n", thisWeight,
                       100. * cs.weight.size() / cs.siz);
            realAddOne(joinID);
        }

        return cs.norm * std::log(1. + cs.f_norm * simSum) - cs.curSum;
    }



    std::chrono::duration<long, std::ratio<1,1000000> > testBrazil(dtype PROP,
                                                                      dtype epsilon = 0.01,
                                                                      int saveWhere=0,
                                                                      int verbose=1
    ) {
        fullCS.clear();
        fullCSWeight.clear();
        std::chrono::duration<long, std::ratio<1, 1000000>> sim_time(0);

        std::vector <idtype> rIDs;
        std::vector <idtype> oDs;
        std::vector <idtype> pIDs;
        std::vector <idtype> rowIDs;
        std::vector <idtype> samplejoinIDs;

        for (int cate = 0; cate <5; cate++) {
            auto st = system_clock::now();
            if (verbose)
                std::cout << "#############       Current category is " << cate << "     ##########\n";


            loadToArr(cate);
            initWeight();

            mallocBrazilnewSim();
            calBrazilnewSim();


            assert(joinNum == orderItemNum);

            if (verbose)std::cout << "join N is " << joinNum << "\n";
            if (verbose)std::cout << "PROP is " << PROP << "\n";

            idtype csSize = (idtype) (PROP * joinNum + 0.5);
            if (verbose)std::cout << "This cate should have [" << csSize << "]\n";


            idtype sampleEachStep = 1. / PROP * std::log(1. / epsilon) + 0.5;


            idtype ano = 1. / PROP * std::log(1. / epsilon) + 0.5;
            if (ano < sampleEachStep)
                sampleEachStep = ano;

            std::cout<<"sample each step is "<<sampleEachStep<<"\n";

            cs.init(joinNum, csSize);
            cs.f_norm = 1. / joinNum;

            auto en = system_clock::now();
            auto duration = duration_cast<microseconds>(en - st);
            sim_time += duration;

            while (csSize--) {
                dtype curMaxBenefit = -1;
                idtype curMaxBenefitID = 0;


                std::vector <idtype> rIDs;
                std::vector <idtype> oIDs;
                std::vector <idtype> pIDs;
                std::vector <idtype> rowIDs;
                std::vector <idtype> samplejoinIDs;

                sampleBatchBrazilnew(sampleEachStep, rIDs, oIDs, pIDs, rowIDs, samplejoinIDs);
                std::vector <dtype> benefit_vec(sampleEachStep);

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < sampleEachStep; i++)
                    benefit_vec[i] = getBenefitBrazilnew(rIDs[i], oIDs[i], pIDs[i], rowIDs[i],samplejoinIDs[i], 0,0);
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
                benefit_vec[i] = getBenefitBrazilnew(rIDs[i], oIDs[i], pIDs[i], rowIDs[i],samplejoinIDs[i], 1, 0);
            }

            freeAll();
            freeBrazilnew();

            fullCS.insert(fullCS.end(), cs.coresetAll.begin(), cs.coresetAll.end());
            fullCSWeight.insert(fullCSWeight.end(), cs.weight.begin(), cs.weight.end());


            if(verbose)std::cout << "Finished!\n";
        }

        if(verbose)printf("Total coreset size 【%d】\n", fullCS.size());

        if(verbose)std::cout <<  "@### 【Similarity】 Spent " << double(sim_time.count()) * microseconds::period::num / microseconds::period::den << " seconds.\n";


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



