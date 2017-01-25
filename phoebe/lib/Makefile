#CXX=icpc
CXX=g++
#CXX=clang++
CXXFLAGS=-O3 -Wall -std=c++11

ifdef PYTHON
CXXFLAGS+=$(shell $(PYTHON)-config --includes)
LDFLAGS+=$(shell $(PYTHON)-config --ldflags)
else
CXXFLAGS+=$(shell python-config --includes)
LDFLAGS+=$(shell python-config --ldflags)
endif

LIBP=libphoebe

all: $(LIBP).so

$(LIBP).so: $(LIBP).cpp *.h
	$(CXX)  -fPIC -shared $(LIBP).cpp -o $(LIBP).so $(CXXFLAGS) $(LDFLAGS)



