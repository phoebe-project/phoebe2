CC=g++
CXXFLAGS=-O3 -Wall -std=c++11

all:gen_roche_area_volume gen_roche_volume

%:%.cpp *.h
	$(CC) $(CXXFLAGS) $*.cpp -o $*

