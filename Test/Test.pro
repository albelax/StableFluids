include(../Common/common.pri)

TARGET = gtest
CONFIG += console c++11
CONFIG -= app_bundle
QT += core

OTHER_FILES += Common

OBJECTS_DIR = $$PWD/obj

INCLUDEPATH+= include /usr/local/include /public/devel/include

LIBS+= -L/usr/local/lib -lgtest -lpthread \
       -L/public/devel/lib/ -lgtest \
       -L$$LIB_INSTALL_DIR -lsolver_cpu -lsolver_gpu \

INCLUDEPATH+= $$PWD/../Common/include \
							 $$INC_INSTALL_DIR \

macx:CONFIG-=app_bundle

QMAKE_RPATHDIR += $$LIB_INSTALL_DIR

HEADERS += include/constructor.h \
					 include/velocity.h \
           include/density.h \
           include/pressure.h \
					 include/divergence.h \
					 $$PWD/../Common/include/*

SOURCES +=$$PWD/src/*.cpp \
					$$PWD/../Common/src/*

