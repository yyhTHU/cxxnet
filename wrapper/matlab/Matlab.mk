MEX=/usr/local/MATLAB/R2014b/bin/mex

all: mex

mex: cxxnet_mex.cpp
	$(MEX) -L../ $+ -lcxxnetwrapper

clean:
	rm -rf cxxnet_mex.m*
