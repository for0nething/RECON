cmake_minimum_required(VERSION 3.5)
project(RECON)

set(CMAKE_CXX_STANDARD 14)

set(LINK_DIR /usr/local/lib/)
set(INC_DIR /usr/local/include/)

include_directories(${INC_DIR})
link_directories(${LINK_DIR})
link_libraries(cnpy)
link_libraries(z)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif()


add_executable(RECON main.cpp)
target_link_libraries(RECON  cnpy z)