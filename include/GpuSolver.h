#ifndef _GPUSOLVER_H
#define _GPUSOLVER_H

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include "Tuple.h"
#include "Solver.h"

// friends the test functions with this class
#define TESTING 1

#if TESTING
#include <gtest/gtest.h>
#endif // TESTING

/// \brief The GpuSolver class. inherits from Solver.h in common, all of it's methods
/// internally make calls to cuda kernels

class GpuSolver : public Solver
{
public:
  GpuSolver();
  ~GpuSolver();

  /// \brief activate allocates memory
  void activate() override;

  /// \brief reset, resets density and velocities
  void reset() override;

  /// \brief cleanBuffer, resets previous density and previous velocities
  void cleanBuffer() override;

  /// \brief setVelBoundary, sets the values at the boundaries
  /// \param flag, 1 for x, 2 for y
  void setVelBoundary( int flag );

  /// \brief setCellBoundary, sets the values at the boundaries for divergence, pressure and density
  /// \param value, either divergence, pressure or density
  void setCellBoundary( real * _value, tuple<unsigned int> & _size );

  /// \brief setVel0, sets the velocity from the mouse input
  /// \param i, destination index
  /// \param j, destination index
  /// \param _vx0, previous mouse position in the x
  /// \param _vy0 previous mouse position in the y
  void setVel0(int i, int j, real _vx0, real _vy0) override;

  /// \brief setD0, sset density from mouse input
  /// \param i, destination index
  /// \param j, destination index
  void setD0(int i, int j ) override;

  /// \brief addSource, adds density and velocity after the user input
  void addSource() override;

  /// \brief animVel, velocity step
  void animVel() override;

  /// \brief animDen, density step
  void animDen() override;

  /// \brief projection, calculates divergence and pressure
  void projection();

  /// \brief advectVelocity, advection of the velocity
  void advectVelocity();

  /// \brief advectCell, advection of the density
  void advectCell();

  /// \brief diffuseVelocity, diffuse velocity
  void diffuseVelocity();

  /// \brief diffuseCell, diffuse Density
  void diffuseCell();
  void exportCSV( std::string _file, tuple<real> * _t, int _sizeX , int _sizeY );

  void gather( real * _value, unsigned int _size );
  void gather2D( real * _value, unsigned int _size );
  void randomizeArrays();
  const real * getDens() override;

#if !TESTING
private:
#endif

  void setParameters();
  void allocateArrays();
  // cpu
  real * m_cpuDensity;
  real * m_cpuPrevDensity;
  tuple<real *> m_cpuPreviousVelocity;
  // end cpu

  void copy( tuple<real> * _src, tuple<real> * _dst, int _size );
  void copy( real * _src, real * _dst, int _size );
  void copy( const real * _src, real * _dst, int _size );
  void copyToDevice( real * _src, real * _dst, int _size );

#if TESTING // I know friend is bad, but it is only to allow tesats to run :(
  FRIEND_TEST( pvx, isEqual );
  FRIEND_TEST( pvy, isEqual );
  FRIEND_TEST( resetDensity, isZero );
  FRIEND_TEST( resetVelocityX, isZero );
  FRIEND_TEST( resetVelocityY, isZero );
  FRIEND_TEST( boundary, velocity_x );
  FRIEND_TEST( boundary, velocity_y );
  FRIEND_TEST( densityBoundary, isEqual );
  FRIEND_TEST( boundary, pressure );
  FRIEND_TEST( divergenceBoundary, isEqual );
  FRIEND_TEST( gather, works );
  FRIEND_TEST( projection, pressure );
  FRIEND_TEST( projection, checkDivergence );
  FRIEND_TEST( projection, velocity_x );
  FRIEND_TEST( projection, velocity_y );
  FRIEND_TEST( advect, velocity_x );
  FRIEND_TEST( advect, velocity_y );
  FRIEND_TEST( velocity_x, diffuse );
  FRIEND_TEST( velocity_y, diffuse );
  FRIEND_TEST( inputVelocity, setVel0 );

#endif // TESTING
};

#endif // _GPUSOLVER_H
