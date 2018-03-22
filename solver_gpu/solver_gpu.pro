# Read shared environment settings from the common include file
include(../Common/common.pri)

# This is set to build a dynamic library
TEMPLATE = lib

# Use this directory to store all the intermediate objects
OBJECTS_DIR = obj

# Set this up as the installation directory for our library
TARGET = $$LIB_INSTALL_DIR/solver_gpu

# Set the C++ flags for this compilation when using the host compiler
QMAKE_CXXFLAGS += -std=c++11 -fPIC

# Directories
INCLUDEPATH += include ${CUDA_PATH}/include \
                       ${CUDA_PATH}/include/cuda \
                       ${CUDA_PATH}/samples/common/inc \
                       $$PWD/../Common \

HEADERS += include/GpuSolver.h \
					 include/rand_gpu.h \
					 include/GpuSolverKernels.cuh

## CUDA_SOURCES - the source (generally .cu) files for nvcc. No spaces in path names
CUDA_SOURCES += cudasrc/GpuSolver.cu \
								cudasrc/rand_gpu.cu \
								cudasrc/GpuSolverKernels.cu


# Link with the following libraries
#LIBS += -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib64/nvidia -lcudadevrt -lcuda -lcudart -lcurand
LIBS += -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib64/nvidia -lcudadevrt \
                                                         -lcuda \
                                                         -lcudart \
                                                         -lcurand \
                                                         -licudata \
                                                         -lcudart_static
 
# CUDA_COMPUTE_ARCH - This will enable nvcc to compiler appropriate architecture specific code for different compute versions
# Set your local CUDA_ARCH environment variable to compile for your particular architecture.
CUDA_COMPUTE_ARCH=${CUDA_ARCH}
isEmpty(CUDA_COMPUTE_ARCH) {
    message(CUDA_COMPUTE_ARCH environment variable not set - set this to your local CUDA compute capability.)
}

# Use the following path for nvcc created object files
CUDA_OBJECTS_DIR = cudaobj
 
# CUDA_DIR - the directory of cuda such that CUDA\<version-number\ contains the bin, lib, src and include folders
# Set this environment variable yourself.
CUDA_DIR=${CUDA_PATH}
isEmpty(CUDA_DIR) {
    message(CUDA_DIR not set - set this to the base directory of your local CUDA install (on the labs this should be /usr))
}
 

## CUDA_INC - all incldues needed by the cuda files (such as CUDA\<version-number\include)
CUDA_INC+= $$join(INCLUDEPATH,' -I','-I',' ')

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = -ccbin $$HOST_COMPILER -m64 -g -G -gencode arch=compute_50,code=sm_50 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_$$CUDA_COMPUTE_ARCH,code=sm_$$CUDA_COMPUTE_ARCH --compiler-options -fno-strict-aliasing --compiler-options -fPIC -use_fast_math --std=c++11 #--ptxas-options=-v

# Define the path and binary for nvcc
NVCCBIN = $$CUDA_DIR/bin/nvcc

#prepare intermediate cuda compiler
cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.o
cuda.commands = $$NVCCBIN $$NVCCFLAGS -dc $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
 
#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cuda.variable_out = CUDA_OBJ
cuda.variable_out += OBJECTS
cuda.clean = $$CUDA_OBJECTS_DIR/*.o
# Note that cuda objects are linked separately into one obj, so these intermediate objects are not included in the final link
cuda.CONFIG = no_link 
QMAKE_EXTRA_COMPILERS += cuda
 
# Prepare the linking compiler step (combine tells us that the compiler will combine all the input files)
cudalink.input = CUDA_OBJ
cudalink.CONFIG = combine
cudalink.output = $$OBJECTS_DIR/cuda_link.o
 
# Tweak arch according to your hw's compute capability

cudalink.commands = $$NVCCBIN $$NVCCFLAGS $$CUDA_INC -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib64/nvidia -lcuda -lcudart -lcudadevrt -lcurand
cudalink.dependency_type = TYPE_C
cudalink.depend_command = $$NVCCBIN $$NVCCFLAGS -M $$CUDA_INC ${QMAKE_FILE_NAME}


# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cudalink

# Set up the post install script to copy the headers into the appropriate directory
includeinstall.commands = mkdir -p $$INC_INSTALL_DIR && cp include/*.h $$INC_INSTALL_DIR
QMAKE_EXTRA_TARGETS += includeinstall
POST_TARGETDEPS += includeinstall
