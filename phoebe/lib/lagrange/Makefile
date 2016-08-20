#CC=g++
CC=clang++
CXXFLAGS=-O3 -Wall -DDEBUG -std=c++11

all: gen_roche_lagrange

%:%.cpp *.h
	$(CC) $(CXXFLAGS) $*.cpp -o $*

