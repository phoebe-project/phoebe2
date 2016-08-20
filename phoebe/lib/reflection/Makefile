#CC=g++
CC=clang++

CXXFLAGS=-O3 -Wall -std=c++11

all: reflection_overcontact reflection_detached reflection_nbody ld_type

%:%.cpp *.h
	$(CC) $(CXXFLAGS) $*.cpp -o $*
