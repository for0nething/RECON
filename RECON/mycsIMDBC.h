#ifndef UNTITLED2_MYCSIMDBC_H
#define UNTITLED2_MYCSIMDBC_H

#include "cnpy.h"
#include "type.h"
#include "util.h"
#include "data.h"

#include <cstring>
#include <fstream>
#include <random>
#include <unordered_map>
#include <chrono>
#include "time.h"
#include "assert.h"

namespace IMDBC {
    using std::chrono::system_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    std::random_device rd;
    std::mt19937 mt(rd());

    cnpy::NpyArray miArr;
    cnpy::NpyArray mixArr;
    cnpy::NpyArray titleArr;
    cnpy::NpyArray nameArr;
    cnpy::NpyArray ciArr;
    cnpy::NpyArray mcArr;
    cnpy::NpyArray mapArr;

    dtype *genders, *countries;
    idtype jN;
    dtype *dp;
    dtype *mi, *mix, *title, *name, *ci, *mc;
    idtype maxMovieID;
    idtype miNum, miDim, mixNum, mixDim, titleNum, titleDim, nameNum, nameDim, ciNum, ciDim, mcNum, mcDim;
    idtype mapNum, mapDim;
    idtype *hashMapV;
    std::unordered_map<idtype, idtype> hashMap;
    dtype *mvSim, *mixSim, *miSim, *titleSim;
    dtype *mRowMap;
    dtype mixWeight, miWeight, titleWeight, personWeight, companyWeight;
    std::discrete_distribution<> movieDis;
    std::vector<idtype> movies;
    std::vector<idtype> movieWeight;
    std::vector<std::vector<idtype> > moviePerson;
    std::vector<std::vector<idtype> > movieCompany;
    std::vector<idtype> constmovieWeight;
    std::vector<idtype> fullCS;
    std::vector<dtype> fullCSWeight;


    cnpy::NpyArray loadNpy(std::string fileDir) {
        cnpy::NpyArray arr = cnpy::npy_load(fileDir);
        return arr;
    }

    void readIMDBCNpy(int cate, int Large = 0, int linear = 0, int cateNum = 10) {
        std::stringstream dir;
        dir.str("");
        if (linear == 0) {
            if (cateNum == 10)
                dir << DATAPATH << (Large ? "IMDBLargeC10" : "IMDBC10") << "-formycs/train-cate-"
                    << cate << "-" << "mi.npy";
            else
                dir << DATAPATH << (Large ? "IMDBLargeC5" : "IMDBC5") << "-formycs/train-cate-"
                    << cate << "-" << "mi.npy";
        } else
            dir << DATAPATH << (Large ? "IMDBLargeCLinearC++" : "IMDBCLinearC++") << "-formycs/train-cate-" << cate
                << "-" << "mi.npy";
        miArr = loadNpy(dir.str());
        dir.str("");
        if (linear == 0) {
            if (cateNum == 10)
                dir << DATAPATH << (Large ? "IMDBLargeC10" : "IMDBC10") << "-formycs/train-cate-"
                    << cate << "-" << "mix.npy";
            else
                dir << DATAPATH << (Large ? "IMDBLargeC5" : "IMDBC5") << "-formycs/train-cate-"
                    << cate << "-" << "mix.npy";
        } else
            dir << DATAPATH << (Large ? "IMDBLargeCLinearC++" : "IMDBCLinearC++") << "-formycs/train-cate-" << cate
                << "-" << "mix.npy";
        mixArr = loadNpy(dir.str());
        dir.str("");
        if (linear == 0) {
            if (cateNum == 10)
                dir << DATAPATH << (Large ? "IMDBLargeC10" : "IMDBC10") << "-formycs/train-cate-"
                    << cate << "-" << "title.npy";
            else
                dir << DATAPATH << (Large ? "IMDBLargeC5" : "IMDBC5") << "-formycs/train-cate-"
                    << cate << "-" << "title.npy";
        } else
            dir << DATAPATH << (Large ? "IMDBLargeCLinearC++" : "IMDBCLinearC++") << "-formycs/train-cate-" << cate
                << "-" << "title.npy";
        titleArr = loadNpy(dir.str());
        dir.str("");
        if (linear == 0) {
            if (cateNum == 10)
                dir << DATAPATH << (Large ? "IMDBLargeC10" : "IMDBC10") << "-formycs/train-cate-"
                    << cate << "-" << "name.npy";
            else
                dir << DATAPATH << (Large ? "IMDBLargeC5" : "IMDBC5") << "-formycs/train-cate-"
                    << cate << "-" << "name.npy";
        } else
            dir << DATAPATH << (Large ? "IMDBLargeCLinearC++" : "IMDBCLinearC++") << "-formycs/train-cate-" << cate
                << "-" << "name.npy";
        nameArr = loadNpy(dir.str());
        dir.str("");
        if (linear == 0) {
            if (cateNum == 10)
                dir << DATAPATH << (Large ? "IMDBLargeC10" : "IMDBC10") << "-formycs/train-cate-"
                    << cate << "-" << "ci.npy";
            else
                dir << DATAPATH << (Large ? "IMDBLargeC5" : "IMDBC5") << "-formycs/train-cate-"
                    << cate << "-" << "ci.npy";
        } else
            dir << DATAPATH << (Large ? "IMDBLargeCLinearC++" : "IMDBCLinearC++") << "-formycs/train-cate-" << cate
                << "-" << "ci.npy";
        ciArr = loadNpy(dir.str());
        dir.str("");
        if (linear == 0) {
            if (cateNum == 10)
                dir << DATAPATH << (Large ? "IMDBLargeC10" : "IMDBC10") << "-formycs/train-cate-"
                    << cate << "-" << "mc.npy";
            else
                dir << DATAPATH << (Large ? "IMDBLargeC5" : "IMDBC5") << "-formycs/train-cate-"
                    << cate << "-" << "mc.npy";
        } else
            dir << DATAPATH << (Large ? "IMDBLargeCLinearC++" : "IMDBCLinearC++") << "-formycs/train-cate-" << cate
                << "-" << "mc.npy";
        mcArr = loadNpy(dir.str());
    }

    void mallocCArray() {
        mRowMap = (dtype *) malloc(100000 * sizeof(dtype));
        genders = (dtype *) malloc(1000000 * sizeof(dtype));
        countries = (dtype *) malloc(1000000 * sizeof(dtype));

        miNum = miArr.shape[0];
        miDim = miArr.shape[1];
        mi = (dtype *) malloc(miNum * miDim * sizeof(dtype));

        mixNum = mixArr.shape[0];
        mixDim = mixArr.shape[1];
        mix = (dtype *) malloc(mixNum * mixDim * sizeof(dtype));
        titleNum = titleArr.shape[0];
        titleDim = titleArr.shape[1];
        title = (dtype *) malloc(titleNum * titleDim * sizeof(dtype));
        nameNum = nameArr.shape[0];
        nameDim = nameArr.shape[1];
        name = (dtype *) malloc(nameNum * nameDim * sizeof(dtype));
        ciNum = ciArr.shape[0];
        ciDim = ciArr.shape[1];
        ci = (dtype *) malloc(ciNum * ciDim * sizeof(dtype));
        mcNum = mcArr.shape[0];
        mcDim = mcArr.shape[1];
        mc = (dtype *) malloc(mcNum * mcDim * sizeof(dtype));
    }

    void loadToArr(int cate, int Large = 0, int linear = 0, int cateNum = 10) {
        readIMDBCNpy(cate, Large, linear, cateNum);
        mallocCArray();
        memcpy(mi, miArr.data<dtype>(), 1LL * miNum * miDim * sizeof(dtype));
        memcpy(mix, mixArr.data<dtype>(), 1LL * mixNum * mixDim * sizeof(dtype));
        memcpy(title, titleArr.data<dtype>(), 1LL * titleNum * titleDim * sizeof(dtype));
        memcpy(name, nameArr.data<dtype>(), 1LL * nameNum * nameDim * sizeof(dtype));
        memcpy(ci, ciArr.data<dtype>(), 1LL * ciNum * ciDim * sizeof(dtype));
        memcpy(mc, mcArr.data<dtype>(), 1LL * mcNum * mcDim * sizeof(dtype));
    }

    void mallocIMDBCSim() {
        mvSim = (dtype *) malloc(titleNum * titleNum * sizeof(dtype));
        mixSim = (dtype *) malloc(titleNum * titleNum * sizeof(dtype));
        miSim = (dtype *) malloc(titleNum * titleNum * sizeof(dtype));
        titleSim = (dtype *) malloc(titleNum * titleNum * sizeof(dtype));
    }

    void calIMDBCSim() {
        initSim(mixSim, mix, mixNum, mixDim, 1, mixDim - 1);
        initSim(miSim, mi, miNum, miDim, 1);
        initSim(titleSim, title, titleNum, titleDim, 1);
        mixWeight = 1.0 / 6;
        miWeight = 1.0 / 6;
        titleWeight = 1.0 / 6;
        personWeight = 1.0 / 2;
        idtype st_id = 0;

        for (idtype i = 0; i < miNum; i++, st_id += miNum) {
            #pragma omp parallel for schedule(static)
            for (idtype j = 0; j < miNum; j++) {
                mvSim[st_id + j] = mixSim[st_id + j] * mixWeight
                                   + miSim[st_id + j] * miWeight
                                   + titleSim[st_id + j] * titleWeight;
            }
        }
    }
    void initWeight(int verbose = 0) {
        maxMovieID = 0;
        jN = 0;
        movies.clear();
        for (idtype i = 0; i < nameNum; i++) {
            idtype pid = name[i * nameDim];
            genders[pid] = name[i * nameDim + 1];
        }
        for (idtype i = 0; i < mcNum; i++) {
            idtype cid = mc[i * mcDim + 1];
            countries[cid] = mc[i * mcDim + 2];
        }
        for (idtype i = 0; i < titleNum; i++) {
            maxMovieID = std::max(maxMovieID, (idtype) title[i * titleDim]);
            mRowMap[(idtype) title[i * titleDim]] = i;
            movies.emplace_back(title[i * titleDim]);
        }
        if (verbose)std::cout << "Max movie ID is " << maxMovieID << "!\n";

        moviePerson.resize(maxMovieID + 1);
        movieCompany.resize(maxMovieID + 1);

        constmovieWeight.clear();
        constmovieWeight.resize(2 * (maxMovieID + 1));
        movieWeight.clear();
        movieWeight.resize(maxMovieID + 1);

        for (idtype i = 0; i <= maxMovieID; i++) {
            moviePerson[i].clear();
            movieCompany[i].clear();
        }

        for (idtype i = 0; i < ciNum; i++) {
            idtype person_id = ci[i * ciDim + 0];
            idtype movie_id = ci[i * ciDim + 1];
            moviePerson[movie_id].emplace_back(person_id);
        }
        if (verbose)std::cout << "moviePerson Weight set finished!\n";

        for (idtype i = 0; i < mcNum; i++) {
            idtype movie_id = mc[i * mcDim + 0];
            idtype company_id = mc[i * mcDim + 1];
            movieCompany[movie_id].emplace_back(company_id);
        }
        if (verbose)std::cout << "movieCompany Weight set finished!\n";
        idtype sm = 0;
        for (idtype i = 0; i <= maxMovieID; i++) {
            movieWeight[i] = (idtype) moviePerson[i].size() * movieCompany[i].size();
            idtype maleCnt = 0, femaleCnt = 0;
            for (auto p:moviePerson[i]) {
                if (genders[p] == 1)++maleCnt;
                else ++femaleCnt;
            }
            constmovieWeight[i << 1] = (idtype) femaleCnt * movieCompany[i].size();
            constmovieWeight[i << 1 | 1] = (idtype) maleCnt * movieCompany[i].size();
            jN += movieWeight[i];
            sm += moviePerson[i].size();
        }
        if (verbose)std::cout << "sm total is " << sm << "\n";
        if (verbose)std::cout << "movie Weight set finished!\n";
        movieDis = std::discrete_distribution<>(movieWeight.begin(), movieWeight.end());
        dp = (dtype *) malloc(3 * (maxMovieID + 1) * sizeof(dtype));
    }

    void sampleOneIMDBC(idtype & m, idtype & p, idtype & c) {
        m = movieDis(mt);
        std::uniform_int_distribution<> personDis = std::uniform_int_distribution<>(0, moviePerson[m].size() - 1);
        p = moviePerson[m][personDis(mt)];
        std::uniform_int_distribution<> companyDis = std::uniform_int_distribution<>(0, movieCompany[m].size() - 1);
        c = movieCompany[m][companyDis(mt)];
    }

    void sampleBatchIMDBC(int sampleSize, std::vector<idtype> &ms, std::vector<idtype> &ps, std::vector<idtype> &cs) {
        for (int i = 0; i < sampleSize; i++)
            sampleOneIMDBC(ms[i], ps[i], cs[i]);
    }

    void realAddOne(idtype m, idtype p, idtype c) {
        --movieWeight[m];
        movieDis = std::discrete_distribution<>(movieWeight.begin(), movieWeight.end());
    }

    void initHashMap(int Large, int linear, int cateNum) {
        std::stringstream dir;
        dir.str("");
        if (linear == 0) {
            if (cateNum == 10)
                dir << DATAPATH << (Large ? "IMDBLargeC10" : "IMDBC10") << "-formycs/idMap.npy";
            else
                dir << DATAPATH << (Large ? "IMDBLargeC5" : "IMDBC5") << "-formycs/idMap.npy";
        } else
            dir << DATAPATH << (Large ? "IMDBLargeCLinearC++" : "IMDBCLinearC++") << "-formycs/idMap.npy";

        mapArr = loadNpy(dir.str());

        mapNum = mapArr.shape[0];
        mapDim = mapArr.shape[1];
        hashMapV = (idtype *) malloc(mapNum * mapDim * sizeof(idtype));

        memcpy(hashMapV, mapArr.data<idtype>(), 1LL * mapNum * mapDim * sizeof(idtype));
        hashMap.clear();
        for (idtype i = 0; i < mapNum; i++) {
            idtype hashV = hashMapV[i * mapDim];
            idtype ID = hashMapV[i * mapDim + 1];
            hashMap[hashV] = ID;
        }

    }

    idtype idInJoin(idtype m, idtype p, idtype c) {
        idtype hashValue = (m + 1) + (p + 1) * 100000LL + (c + 1) * 100000000000LL;
        assert(hashMap.find(hashValue) != hashMap.end());
        return hashMap[hashValue];
    }


    dtype getBenefitIMDBC(idtype m, idtype p, idtype c, bool change = true, int verbose = 1) {

        dtype simSum = 0;
        dtype thisWeight = 0.;

        idtype gender = genders[p];
        dtype country = countries[c];

        idtype mRowID = mRowMap[m];
        assert((idtype) title[mRowID * titleDim] == m);

        idtype mSt = mRowID * titleNum;
        for (idtype i: movies) {
            idtype iRowID = mRowMap[i];
            assert((idtype) title[iRowID * titleDim] == i);

            dtype newSim = mvSim[mSt + iRowID];
            bool addCompanyDiff = false;
            for (auto c_: movieCompany[m])
                if (countries[c_] != country) {
                    addCompanyDiff = true;
                    break;
                }
            if (!addCompanyDiff)
                newSim += companyWeight;

            dtype maleSim = ((gender == 1) ? personWeight : 0) + newSim;
            dtype femaleSim = ((gender == 0) ? personWeight : 0) + newSim;

            simSum += std::max(dp[i << 1 | 1], maleSim) * constmovieWeight[i << 1 | 1];
            simSum += std::max(dp[i << 1], femaleSim) * constmovieWeight[i << 1];

            if (maleSim > dp[i << 1 | 1] && change) {
                dp[i << 1 | 1] = maleSim;
                if (cs.nn[i << 1 | 1] != -1) {
                    cs.weight[cs.nn[i << 1 | 1]] -= constmovieWeight[i << 1 | 1]; //
                }
                cs.nn[i << 1 | 1] = cs.weight.size();
                thisWeight += constmovieWeight[i << 1 | 1];
            }

            if (femaleSim > dp[i << 1] && change) {
                dp[i << 1] = femaleSim;
                if (cs.nn[i << 1] != -1) {
                    cs.weight[cs.nn[i << 1]] -= constmovieWeight[i << 1]; //
                }
                cs.nn[i << 1] = cs.weight.size();
                thisWeight += constmovieWeight[i << 1];
            }

        }
        if (change) {
            cs.curSum = simSum;
            cs.curSum = cs.norm * std::log(1. + cs.f_norm * cs.curSum);
            cs.add(idInJoin(m, p, c));
            cs.weight.emplace_back(thisWeight);
            if (verbose)
                printf("    add this weight is %.2f         Current progress 【%.2f %%】\n", thisWeight,
                       100. * cs.weight.size() / cs.siz);
            realAddOne(m, p, c);
        }
        return cs.norm * std::log(1. + cs.f_norm * simSum) - cs.curSum;
    }

    std::chrono::duration<long, std::ratio<1, 1000000>> testIMDBC(dtype PROP,
                                                                  idtype Large = 0,
                                                                  dtype epsilon = 0.01,
                                                                  int linear = 0,
                                                                  int cateNum = 10,
                                                                  int saveWhere = 0,
                                                                  int verbose = 1,
                                                                  int assignSampleSize = 0
    ) {
        fullCS.clear();
        fullCSWeight.clear();

        std::chrono::duration<long, std::ratio<1, 1000000>> sim_time(0);
        auto st = system_clock::now();
        initHashMap(Large, linear, cateNum);
        auto en = system_clock::now();
        auto duration = duration_cast<microseconds>(en - st);
        sim_time += duration;

        for (int cate = 0; cate < (linear == 0 ? cateNum : 87); cate++) {
            st = system_clock::now();
            if (verbose)std::cout << "#############       Current category is " << cate << "     ##########\n";

            loadToArr(cate, Large, linear, cateNum);
            if (verbose)std::cout << "title num is " << titleNum << "\n";

            initWeight(verbose);
            mallocIMDBCSim();
            calIMDBCSim();

            if (verbose)std::cout << "join N is " << jN << "\n";
            if (verbose)std::cout << "PROP is " << PROP << "\n";
            idtype csSize = (idtype) (PROP * jN);

            if (verbose)std::cout << "This cate should have [" << csSize << "]\n";
            idtype sampleEachStep = 500;

            en = system_clock::now();
            duration = duration_cast<microseconds>(en - st);
            sim_time += duration;
            std::vector<idtype> Ms(sampleEachStep), Ps(sampleEachStep), Cs(sampleEachStep);


            cs.init(2 * (maxMovieID + 1), csSize);
            cs.f_norm = 1. / jN;

            if (verbose)
                std::cout << "company weight is " << companyWeight << " person weight is " << personWeight << "\n";

            while (csSize--) {
                dtype curMaxBenefit = -1;
                idtype curMaxBenefitID = 0;

                sampleBatchIMDBC(sampleEachStep, Ms, Ps, Cs);

                std::vector<dtype> benefit_vec(sampleEachStep);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < sampleEachStep; i++)
                    benefit_vec[i] = getBenefitIMDBC(Ms[i], Ps[i], Cs[i], false);

                idtype i = 0;
                for (auto val : benefit_vec) {
                    if (val > curMaxBenefit) {
                        curMaxBenefit = val;
                        curMaxBenefitID = i;
                    }
                    ++i;
                }
                i = curMaxBenefitID;
                if (verbose)std::cout << "Benefit is " << curMaxBenefit;
                getBenefitIMDBC(Ms[i], Ps[i], Cs[i], true, verbose);

            }

            fullCS.insert(fullCS.end(), cs.coresetAll.begin(), cs.coresetAll.end());
            fullCSWeight.insert(fullCSWeight.end(), cs.weight.begin(), cs.weight.end());

        }
        printf("Total coreset size 【%d】\n", fullCS.size());

        std::cout << "@### 【Similarity】 Spent "
                  << double(sim_time.count()) * microseconds::period::num / microseconds::period::den << " seconds.\n";

        assert(saveWhere==0);
        if (!saveWhere) {
            std::stringstream dir;
            dir.str("");
            if (linear == 0) {
                if (cateNum == 10)
                    dir << CSPATH << (Large ? "IMDBLargeC10" : "IMDBC10");
                else
                    dir << CSPATH << (Large ? "IMDBLargeC5" : "IMDBC5");
            } else
                dir << CSPATH << (Large ? "IMDBCLinear" : "IMDBLargeCLinear");
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
#endif //UNTITLED2_MYCSIMDBC_H



