#ifndef _GPUSOLVER_H
#define _GPUSOLVER_H

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include "tuple.h"

#define SWAP(value0,value) { float *tmp = value0; value0 = value; value = tmp; }
#define TESTING 1

#if TESTING
#include <gtest/gtest.h>
//#define private public
#endif // TESTING

class GpuSolver
{
public:
  GpuSolver();
  ~GpuSolver();
  void init();
  void cleanBuffer();
  int vxIdx(int i, int j){ return j*m_rowVelocity.x+i; }
  int vyIdx(int i, int j){ return j*m_rowVelocity.y+i; }
  int cIdx(int i, int j){ return j*m_gridSize.x+i; }
  void exportCSV(std::string _file, tuple<float> * _t, int _sizeX , int _sizeY );

private:
  void setParameters();
  void allocateArrays();
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
  void copy( tuple<float> * _src, tuple<float> * _dst, int _size );

#if TESTING
//  tuple<float> * m_pvxHost;


  // I know friend is bad, but it is only to allow tesats to run :(
  FRIEND_TEST(pvx, isEqual);
#endif // TESTING
};

#endif // _GPUSOLVER_H
