include(../Common/common.pri)

TARGET = gtest
CONFIG += console c++11
CONFIG -= app_bundle
QT += core

OTHER_FILES += Common

OBJECTS_DIR = $$PWD/obj

INCLUDEPATH+= include /usr/local/include /public/devel/include

LIBS+= -L/usr/local/lib -lgtest -lpthread \
       -L$$LIB_INSTALL_DIR -lsolver_cpu -lsolver_gpu \

INCLUDEPATH+= $$PWD/../Common/ \
							 $$INC_INSTALL_DIR \

macx:CONFIG-=app_bundle

QMAKE_RPATHDIR += $$LIB_INSTALL_DIR

HEADERS += include/constructor.h \
					 include/velocity.h
SOURCES +=$$PWD/src/*.cpp
