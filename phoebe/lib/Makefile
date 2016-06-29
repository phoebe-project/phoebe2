CC=g++
CXXFLAGS=-O3 -Wall -std=c++11
PYTHON=-I/usr/include/python2.7/ -lpython2.7

LIBP=libphoebe

all:omega_crit points_on_x_axis pole rot_star $(LIBP).so


$(LIBP).so: $(LIBP).cpp
	$(CC)  -fPIC -shared $(LIBP).cpp -o $(LIBP).so $(CXXFLAGS) $(PYTHON)

%:%.cpp *.h
	$(CC) $(CXXFLAGS) $*.cpp -o $*



