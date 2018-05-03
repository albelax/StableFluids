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
<p align="center">
### 64x64
| Benchmark             |        Time   |     CPU      |   Iterations | 
| ----------------------|---------------|--------------|--------------| 
| Creation_of_a_string  |        0 ns   |        0 ns  |  1000000000  | 
| CPU_solverCreation    |       24 ns   |       24 ns  |    31348493  | 
| GPU_solverCreation    |       24 ns   |       24 ns  |    28930681  | 
| CPU_solverActivation  |   102237 ns   |   102229 ns  |        6826  | 
| GPU_solverActivation  |  1122034 ns   |  1115693 ns  |         636  | 
| CPU_projection        |  1687347 ns   |  1687212 ns  |         414  | 
| GPU_projection        |  1811452 ns   |  1811300 ns  |         478  | 
| CPU_advectVelocity    |   433016 ns   |   432981 ns  |        1620  | 
| GPU_advectVelocity    |   232544 ns   |   232526 ns  |        4207  | 
| CPU_advectCell        |   318658 ns   |   318634 ns  |        2111  | 
| GPU_advectCell        |   117705 ns   |   117690 ns  |       11662  | 
| CPU_diffuseVelocity   |  3216789 ns   |  3216502 ns  |         217  | 
| GPU_diffuseVelocity   |  2222757 ns   |  2222577 ns  |        1000  | 
| CPU_diffuseDensity    |  1488659 ns   |  1488532 ns  |         469  | 
| GPU_diffuseDensity    |  3024257 ns   |  3024020 ns  |       10000  | 
| CPU_animateVelocity   |  3846763 ns   |  3846440 ns  |         181  | 
| GPU_animateVelocity   |  3816887 ns   |  3816581 ns  |         195  | 
| CPU_animateDensity    |   334304 ns   |   334278 ns  |        2270  | 
| GPU_animateDensity    |   117858 ns   |   117849 ns  |       11661  | 
| CPU_addSource         |    37362 ns   |    37359 ns  |       18710  | 
| GPU_addSource         |   348491 ns   |   348081 ns  |        2053  | 

<\p>
