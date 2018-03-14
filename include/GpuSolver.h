#ifndef _GPUSOLVER_H
#define _GPUSOLVER_H

#include <stdio.h>
#include <time.h>
#include <iostream>

#define SWAP(value0,value) { float *tmp = value0; value0 = value; value = tmp; }


template <class T>
class tuple
{
// I genuinly just wanted a generic struct...
public:
    tuple() = default;
    tuple( T _x, T _y ) { x = _x; y = _y; }
    T x;
    T y;
};

class GpuSolver
{
public:
    GpuSolver();
    ~GpuSolver() = default;
    void init();
    void cleanBuffer();
    int vxIdx(int i, int j){ return j*m_rowVelocity.x+i; }
    int vyIdx(int i, int j){ return j*m_rowVelocity.y+i; }
    int cIdx(int i, int j){ return j*m_gridSize.x+i; }
private:
    int m_totCell;
    int m_totVelX;
    int m_totVelY;
    float m_timeStep;
    float m_diffusion;
    float m_viscosity;

    float * m_density;
    float * m_previousDensity; // d0
    float * m_divergence;
    float * m_pressure;
    tuple<int> m_gridSize;
    tuple<int> m_rowVelocity;
    tuple<int> m_columnVelocity;
    tuple<float> m_min;
    tuple<float> m_max;
    /// \brief velocity, stores to pointers to chunks of memory storing the velocities in x and y
    tuple<float *> m_velocity;
    tuple<float *> m_previousVelocity;
    tuple<float> * m_pvx;
    tuple<float> * m_pvy;
//    thrust::device_vector<float> d_px; // here just to remind me that I can use thrust here as well :)
};

#endif // _GPUSOLVER_H
