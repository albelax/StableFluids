# Read shared environment settings from the common include file
include(../Common/common.pri)

# Use this setting to build a shared lib (add staticlib to CONFIG if you want one)
TEMPLATE = lib

# Flags for compilation
QMAKE_CXXFLAGS += -std=c++11 -fPIC -Wall -Wextra -pedantic

# Use this directory to store all the intermediate objects
OBJECTS_DIR = obj

# This sets the build target directory
TARGET = $$LIB_INSTALL_DIR/solver_cpu

#LIBS += -L/public/devel/lib -lbenchmark

# Include headers
HEADERS += include/*.h \
					 $$PWD/../Common/*.h

# Include source files
SOURCES += src/*.cpp

# Set up the include path
INCLUDEPATH += include \
							 $$PWD/../Common \
               $$PWD/../Common/glm \

# Set up the post install script to copy the headers into the appropriate directory
includeinstall.commands = mkdir -p $$INC_INSTALL_DIR && cp include/*.h $$INC_INSTALL_DIR
QMAKE_EXTRA_TARGETS += includeinstall
POST_TARGETDEPS += includeinstall
