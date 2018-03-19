include(../Common/common.pri)

TARGET = Benchmark
CONFIG += console c++11
CONFIG -= app_bundle
QT += core

OTHER_FILES += Common

SOURCES += $$PWD/src/*.cpp
#HEADERS +=$$PWD/include/*.h
OBJECTS_DIR = $$PWD/obj

INCLUDEPATH+= /usr/local/include /public/devel/include
LIBS+= -L/usr/local/lib -lgtest -lpthread \
       -L$$LIB_INSTALL_DIR -lsolver_cpu -lsolver_gpu \
       -L/public/devel/lib -lbenchmark


INCLUDEPATH+= $$PWD/../Common/ \
               $$INC_INSTALL_DIR \


macx:CONFIG-=app_bundle

QMAKE_RPATHDIR += $$LIB_INSTALL_DIR
