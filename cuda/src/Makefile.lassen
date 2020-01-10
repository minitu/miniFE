#-----------------------------------------------------------------------

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

CFLAGS = -O3
CXXFLAGS = -O3

# For debugging, the macro MINIFE_DEBUG will cause miniFE to dump a log file
# from each proc containing various information.
# This macro will also enable a somewhat expensive range-check on indices in
# the exchange_externals function.

DUMPI_HOME=$(HOME)/sst-dumpi/install

#CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) -DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK  #-DMINIFE_DEBUG
CPPFLAGS = -std=c++11 -g -I. -I../utils -I../fem -I$(DUMPI_HOME)/include $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) -DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK -DMEASURE_TIME #-DDUMPI_TRACE  #-DMATVEC_OVERLAP #-DGPUDIRECT #-DMINIFE_DEBUG
#CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) #-DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK -DMATVEC_OVERLAP #-DGPUDIRECT #-DMATVEC_OVERLAP #-DMINIFE_DEBUG
#CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) -DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK -DGPUDIRECT -DMATVEC_OVERLAP #-DMINIFE_DEBUG

LDFLAGS = -g -L${CUDA_HOME}/lib64 -fopenmp -L$(DUMPI_HOME)/lib #-L${PPT_ROOT}/lib64 -lmpitrace -lotf2trace #-Wl,--hash-style=sysv -Wl,--emit-stub-syms
LIBS = -ldl -lm -lrt -lnsl -lutil -lnvToolsExt -lcudart -ldumpi

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

#mvapich
#MPICFLAGS=-I/usr/local/mvapich/include -pthread
#MPILDFLAGS=-L/usr/local/mvapich/lib -lmpichcxx -lmpich -lopa -lmpl -lcudart -lcuda -libverbs -ldl -lrt -lm -lpthread
#c++ -L/usr/local/cuda//lib64 -L/usr/local/cuda//lib -L/lib -L/lib -L/lib -Wl,-rpath,/lib -L/lib -Wl,-rpath,/lib -L/lib -Wl,-rpath,/lib -L/lib -L/lib -I/usr/local/mvapich/include -L/usr/local/mvapich/lib -lmpichcxx -lmpich -lopa -lmpl -lcudart -lcuda -libverbs -ldl -lrt -lm -lpthread

#openmpi
MPI_HOME=${MPI_ROOT}
MPICFLAGS=-I$(MPI_HOME)/include -pthread

NVCCFLAGS=-lineinfo -gencode=arch=compute_70,code=\"sm_70,compute_70\"

NVCC=nvcc -Xcompiler -fopenmp -Xptxas -dlcm=cg
CXX=mpicxx
CC=mpicc

include make_targets
