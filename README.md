# Stable Fluids
<p align="center">
  <img width="256" height="256" src="README_IMAGES/smoke.gif">
</p>


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
The project is divided in:

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

## The GPU Solver

### Structure
The solver inherits from Solver.h, in the common folder, the class is defined in the [GpuSolver.h](https://github.com/albelax/StableFluids/blob/master/solver_gpu/include/GpuSolver.h) and the class is implemented in [GpuSolver.cu](https://github.com/albelax/StableFluids/blob/master/solver_gpu/cudasrc/GpuSolver.cu), which looks like a normal c++ class, however all the functionalities are implemented in [GPUSolverKernels.cu] (https://github.com/albelax/StableFluids/blob/master/solver_gpu/cudasrc/GpuSolverKernels.cu), this meant that I could write cuda kernels and wrap them up in methods of a class, allowing me to take advantage of CUDA's performance while keeping a high level interface.

### Micro Optimisations
Moving the solver from serial to Parallel provided a significant speedup, however I decided to adapt a few techniques to reduce hoverheads whenever possible:
* Constant memory, something similar to an L2 cache, slower than L1 cache but way faster than global memory, I decided to store data that was commonly used in global memory so I could avoid launching a kernel with the same data every time.

* Streams, usually kernels are launched asyncronously, but once launched they get queued up and they execute one at the time,
whenever possible I used multiple streams to launch and execute kernels in parallel, of course this was rarely possible as most of the components are interdependent.

### Future Improvements


## Results

### 64x64

|      Benchmark       |      Time      |     CPU       | Iterations |
| ---------------------|----------------|---------------|------------|
| CPU_solverCreation   |        17 ns   |        17 ns  |   40403571 |
| GPU_solverCreation   |        19 ns   |        19 ns  |   36194377 |
| CPU_solverActivation |     80724 ns   |     80697 ns  |       8521 |
| GPU_solverActivation |    649797 ns   |    644707 ns  |       1157 |
| CPU_projection       |   1365934 ns   |   1365945 ns  |        513 |
| GPU_projection       |    547331 ns   |    547332 ns  |       1000 |
| CPU_advectVelocity   |    360546 ns   |    360548 ns  |       1767 |
| GPU_advectVelocity   |     68330 ns   |     68331 ns  |      10000 |
| CPU_advectCell       |    240370 ns   |    240371 ns  |       2818 |
| GPU_advectCell       |     31872 ns   |     31873 ns  |      22616 |
| CPU_diffuseVelocity  |   2665851 ns   |   2665869 ns  |        263 |
| GPU_diffuseVelocity  |    595574 ns   |    595581 ns  |       1000 |
| CPU_diffuseDensity   |   1171699 ns   |   1171712 ns  |        597 |
| GPU_diffuseDensity   |    770841 ns   |    770832 ns  |      10000 |
| CPU_animateVelocity  |   3079318 ns   |   3078935 ns  |        229 |
| GPU_animateVelocity  |   1119373 ns   |   1119329 ns  |        685 |
| CPU_animateDensity   |    240566 ns   |    240569 ns  |       2912 |
| GPU_animateDensity   |     31930 ns   |     31929 ns  |      22686 |
| CPU_addSource        |     28556 ns   |     28555 ns  |      24572 |
| GPU_addSource        |    147887 ns   |    147715 ns  |       4914 |

###128

|      Benchmark         |       Time       |      CPU      |  Iterations  |
|------------------------|----------------- |---------------|--------------|
|  Creation_of_a_string  |          4 ns    |         4 ns  |   196118568  |
|  CPU_solverCreation    |         17 ns    |        17 ns  |    40455472  |
|  GPU_solverCreation    |         19 ns    |        19 ns  |    36708187  |
|  CPU_solverActivation  |     298321 ns    |    298324 ns  |        2349  |
|  GPU_solverActivation  |     667458 ns    |    663841 ns  |        1123  |
|  CPU_projection        |    6713939 ns    |   6700848 ns  |         105  |
|  GPU_projection        |     605262 ns    |    603770 ns  |        1319  |
|  CPU_advectVelocity    |    1451872 ns    |   1446323 ns  |         484  |
|  GPU_advectVelocity    |      80009 ns    |     79694 ns  |       12957  |
|  CPU_advectCell        |    1020521 ns    |   1016595 ns  |         695  |
|  GPU_advectCell        |      39250 ns    |     39095 ns  |       18304  |
|  CPU_diffuseVelocity   |   10697569 ns    |  10656041 ns  |          65  |
|  GPU_diffuseVelocity   |     884542 ns    |    880996 ns  |        1000  |
|  CPU_diffuseDensity    |    5091461 ns    |   5071702 ns  |         140  |
|  GPU_diffuseDensity    |     620473 ns    |    618055 ns  |        1545  |
|  CPU_animateVelocity   |   23396999 ns    |  23305178 ns  |          34  |
|  GPU_animateVelocity   |    1287062 ns    |   1282012 ns  |         601  |
|  CPU_animateDensity    |    1022019 ns    |   1018040 ns  |         695  |
|  GPU_animateDensity    |      39247 ns    |     39088 ns  |       18368  |
|  CPU_addSource         |     108270 ns    |    107843 ns  |        6476  |
|  GPU_addSource         |     205904 ns    |    205107 ns  |        3409  |
