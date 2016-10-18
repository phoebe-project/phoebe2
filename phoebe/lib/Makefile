#CXX=icpc
CXX=g++
#CXX=clang++
CXXFLAGS=-O3 -Wall -std=c++11
PYTHON=-I/usr/include/python2.7/ -lpython2.7

LIBP=libphoebe

all: $(LIBP).so

$(LIBP).so: $(LIBP).cpp *.h
	$(CXX)  -fPIC -shared $(LIBP).cpp -o $(LIBP).so $(CXXFLAGS) $(PYTHON)


