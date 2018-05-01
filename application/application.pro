include(../Common/common.pri)

QT += core gui
CONFIG += console c++11

TEMPLATE = app
TARGET=$$BIN_INSTALL_DIR/application

QT += opengl \
      core \
      gui

CONFIG += console \
          c++11

CONFIG -= app_bundle

INCLUDEPATH += include \
               ui \
							 $$PWD/../Common/include \
							 $$PWD/../Common/glm \
							 $$INC_INSTALL_DIR

HEADERS += include/MainWindow.h \
           include/GLWindow.h \
           include/Camera.h \
           include/TrackballCamera.h \
           include/Shader.h \
					 include/Mesh.h \
					 $$PWD/../Common/include/*


SOURCES += src/main.cpp \
           src/MainWindow.cpp \
           src/GLWindow.cpp \
           src/Camera.cpp \
           src/TrackballCamera.cpp \
           src/Shader.cpp \
					 src/Mesh.cpp \
					 $$PWD/../Common/src/*

OTHER_FILES += shaders/* \
               models/*


FORMS += ui/mainwindow.ui

UI_HEADERS_DIR = ui
OBJECTS_DIR = obj
MOC_DIR = moc
UI_DIR = ui

QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic

LIBS += -L$$LIB_INSTALL_DIR -lsolver_cpu -lsolver_gpu
linux:LIBS += -lGL -lGLU -lGLEW

QMAKE_RPATHDIR += $$LIB_INSTALL_DIR
