#include <iostream>
#include <queue>
#include <cstring>
#include "type.h"
#include "util.h"
#include "data.h"
#include "omp.h"
#include "cnpy.h"
#include <cstdio>
#include<random>
#include<algorithm>
#include <fstream>
#include <map>
#include <time.h>
#include "mycsIMDBC.h"
#include "mycsStackn.h"
#include "mycsTaxi.h"
#include "mycsBrazil.h"
#include <chrono>

using namespace  std::chrono;
using std::chrono::system_clock;


int main(int argc, char** argv) {

    omp_set_num_threads(tc);

    std::cout<<argv[1]<<"\n";
    std::string dataName = (std::string)argv[1];
    auto st   = system_clock::now();

    std::chrono::duration<long, std::ratio<1, 1000000>> sim_time(0);

    if(dataName == "IMDB")
        sim_time = IMDBC::testIMDBC(std::stod(argv[2]), std::atol(argv[3]),0.01, std::atol(argv[4]));
    else if(dataName == "stack")
        sim_time = stackn::testStackn(std::stod(argv[2]));
    else if(dataName == "Brazil")
        sim_time = Brazil::testBrazil(std::stod(argv[2]));
    else if(dataName == "taxi")
        sim_time = taxi::testTaxi(std::stod(argv[2]));


    auto en = system_clock::now();
    auto duration = duration_cast<microseconds>(en - st);
    std::cout << "### Find Coreset Spent "
              << double(duration.count()) * microseconds::period::num / microseconds::period::den << " seconds.\n";
    std::cout << "### Find Coreset(except sim) Spent "
              << double((duration - sim_time).count()) * microseconds::period::num / microseconds::period::den
              << " seconds.\n";
}



