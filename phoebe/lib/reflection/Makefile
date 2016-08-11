CC=g++
CXXFLAGS=-O3 -Wall -std=c++11

all: reflection_overcontact reflection_detached ld_type

%:%.cpp *.h
	$(CC) $(CXXFLAGS) $*.cpp -o $*
