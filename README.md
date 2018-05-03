# Stable Fluids
Parallel implementation of Jos Stam's Stable Fluids

## Project Overview
This Project was developed as a third year programming assignment,
The brief required to get a working algorithm and write it in parallel using CUDA.
I chose the [original solver](https://github.com/finallyjustice/stablefluids) 
because it was a really efficient and performant implemtation that uses a Data Oriented approach.

### Configuration
The following environment variables are needed to compile this out of the box:
* CUDA_PATH : Needs to point to the base directory of your cuda lib and includes
* CUDA_ARCH : Your local device compute (e.g. 30 or 52 or whatever)
* HOST_COMPILER : Your local g++ compiler (compatible with cuda compiles, g++ 4.9)

In order to compile and run the application:
* Qmake
* make
* run $$PWD/bin/application

### Dependencies
* [Qt]( https://www.qt.io/ ) - QtCreator was my IDE of choice, I also used [QWidget](http://doc.qt.io/qt-5/qwidget.html) and [QImage](http://doc.qt.io/qt-5/qimage.html)
* [CUDA](https://developer.nvidia.com/cuda-toolkit) - used to accelerate the Solver in parallel
* [OpenGL](https://www.opengl.org/) ( 4.0 ) - used to draw the smoke simulation
* [Google Test](https://github.com/google/googletest) - used to test the correctness of the implementation
* [Google Benchmark](https://github.com/google/benchmark) - used to measure the speedups

### Project Structure
The project is divided in a parts:

* solver_cpu: the serial implementation of the solver, slightly changed from the original version,
the subProject compiles into a shared library

* solver_gpu: the parallel version of the solver, 
the subProject compiles into a shared library 

* application: The OpenGL project that uses the two libraries and runs the simulation

* Test: This project's sole purpose is to check that every component works correctly

* Benchmark: This small project is used to measure the speedups

##### Common
Common contains libraries and headers that the solvers and the applications need:
* [glm](https://glm.g-truc.net/0.9.8/index.html) - used in the aplication
* [parameters](https://github.com/albelax/StableFluids/blob/master/Common/include/parameters.h) - used to tweak the parameters of both solvers
* [Solver](https://github.com/albelax/StableFluids/blob/master/Common/include/Solver.h) - base class for both solvers, perhaps not the best decision but made it easy to swap between the two from the application side
* [tuple](https://github.com/albelax/StableFluids/blob/master/Common/include/tuple.h) - a simple generic container, also defines the type "real", used to swap easyly between floats and doubles

## Workflow
### Analysis - Profiling
The first task of the project was to detect the most expensive components of the solver, I did that using [Callgrind](http://valgrind.org/docs/manual/cl-manual.html) embedded in QtCreator

![Callgrind](README_IMAGES/Callgrind.png)

As showed in the image above the most expensive part of the solver was the velocity step ( animVel ), the second most expensive call is the projection, which caluclates the pressure,
and it's run two times in the velocity step, so I knew that the pressure was going to be the most important function to speed up on the GPU.

### Implementation
Due to the nature of the project I knew I had to define my own workflow to minimize errors and make sure I was proceeding in the right direction.
My approach similar to [test driven development](https://en.wikipedia.org/wiki/Test-driven_development), 
I would write the test before implementing new components, once implemented, tested against the original solver, and made sure the test passed I would benchmark that component.

