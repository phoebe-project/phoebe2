#CC=g++
CC=clang++
CXXFLAGS=-O3 -Wall -std=c++11

all:triang_marching sphere_area_volume sphere_test_case horizon mesh_offseting

%:%.cpp *.h
	$(CC) $(CXXFLAGS) $*.cpp -o $*

