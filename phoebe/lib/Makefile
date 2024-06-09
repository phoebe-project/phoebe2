#CXX=icpc
CXX=g++
#CXX=clang++
CXXFLAGS=-O3 -Wall -Wextra

ifdef PYTHON
CXXFLAGS+=$(shell $(PYTHON)-config --includes)
LDFLAGS+=$(shell $(PYTHON)-config --ldflags)
else
CXXFLAGS+=$(shell python3-config --includes)
LDFLAGS+=$(shell python3-config --ldflags)
endif

LIBP=libphoebe

all: $(LIBP).so

$(LIBP).so: $(LIBP).cpp *.h
	$(CXX)  -fPIC -shared $(LIBP).cpp -o $(LIBP).so $(CXXFLAGS) $(LDFLAGS)



