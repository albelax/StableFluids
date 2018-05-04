/** File:    MacStableSolver.h
 ** Author:  Dongli Zhang
 ** Contact: dongli.zhang0129@gmail.com
 **
 ** Copyright (C) Dongli Zhang 2013
 **
 ** This program is free software;  you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation; either version 2 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY;  without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
 ** the GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program;  if not, write to the Free Software
 ** Foundation, 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef __MACSTABLESOLVER_H__
#define __MACSTABLESOLVER_H__
#include <string.h>
#include <stdio.h>
#include <iostream>
#include "Tuple.h"
#include "rand_cpu.h"
#include <QImage>
#include "Solver.h"

// friends the test functions with this class
#define TESTING 1

#if TESTING
#include <gtest/gtest.h>
#endif // TESTING

class StableSolverCpu : public Solver
{
public:
  StableSolverCpu();
  ~StableSolverCpu();
  /// \brief activate allocates memory
  void activate() override;

  /// \brief reset, resets density and velocities
  void reset() override;

  /// \brief cleanBuffer, resets previous density and previous velocities
  void cleanBuffer() override;

  /// \brief setVelBoundary, sets the values at the boundaries
  /// \param flag, 1 for x, 2 for y
  void setVelBoundary(int flag);

  /// \brief setCellBoundary, sets the values at the boundaries for divergence, pressure and density
  /// \param value, either divergence, pressure or density
  void setCellBoundary(real *value);

  /// \brief projection, calculates divergence and pressure
  void projection();

  /// \brief advectVel, advection of the velocity
  void advectVel();

  /// \brief advectCell, advection of the density
  /// \param value, current density
  /// \param value0, previous density
  void advectCell(real *value, real *value0);

  /// \brief diffuseVel, diffuse velocity
  void diffuseVel();

  /// \brief diffuseCell, diffuse Density
  /// \param value, current density
  /// \param value0, previous density
  void diffuseCell(real *value, real *value0);

  /// \brief addSource, adds density and velocity after the user input
  void addSource() override;

  /// \brief animVel, velocity step
  void animVel() override;

  /// \brief animDen, density step
  void animDen() override;

  /// \brief getVX, returns velocity in the x
  real* getVX() const { return m_velocity.x; }

  /// \brief getVY, returns velocity in the y
  real* getVY() const { return m_velocity.y; }

  tuple<real> * getPVX() const { return m_pvx; }
  tuple<real> * getPVY() const { return m_pvy; }

  /// \brief getDens, returns the density
  const real * getDens() override { return m_density; }

  /// \brief getCellVel, returns a tuple containing the velocity in the x and y at the requested index
  /// \param i, index in the x
  /// \param j, index in the y
  tuple<real> getCellVel(int i, int j) const
  {
    real x = (m_velocity.x[vxIdx(i, j)]+m_velocity.x[vxIdx(i+1, j)]) / 2.0f;
    real y = (m_velocity.y[vyIdx(i, j)]+m_velocity.y[vyIdx(i, j+1)]) / 2.0f;
    tuple<real> ret( x, y );
    return ret;
  }

  /// \brief getDens, calculates density of the cell[i][j]
  real getDens(int i, int j) const
  {
    real dens = m_density[cIdx(i-1, j-1)] + m_density[cIdx(i, j-1)] +
        m_density[cIdx(i-1, j)] + m_density[cIdx(i, j)];
    return dens / 4.0f;
  }

  /// \brief setVel0, sets the velocity from the mouse input
  /// \param i, destination index
  /// \param j, destination index
  /// \param _vx0, previous mouse position in the x
  /// \param _vy0 previous mouse position in the y
  void setVel0(int i, int j, real _vx0, real _vy0)
  {
    m_previousVelocity.x[vxIdx(i, j)] = _vx0;
    m_previousVelocity.x[vxIdx(i+1, j)] = _vx0;
    m_previousVelocity.y[vyIdx(i, j)] = _vy0;
    m_previousVelocity.y[vyIdx(i, j+1)] = _vy0;
  }

  /// \brief setD0, sset density from mouse input
  /// \param i, destination index
  /// \param j, destination index
  void setD0(int i, int j ){ m_previousDensity[cIdx(i, j)] = m_inputDensity; }

  /// \brief exportCSV, small csv exporter, useful to debug
  /// \param _file, name of output file
  /// \param _t, tuples in input
  /// \param _sizeX, width
  /// \param _sizeY, height
  void exportCSV( std::string _file, tuple<real> * _t, int _sizeX, int _sizeY );

  /// \brief randomizeArrays, random generator, useful for tests
  void randomizeArrays();

#if TESTING
  FRIEND_TEST( pvx, isEqual );
  FRIEND_TEST( pvy, isEqual );
  FRIEND_TEST( resetDensity, isEqual );
  FRIEND_TEST( boundary, velocity_x );
  FRIEND_TEST( boundary, velocity_y );
  FRIEND_TEST( densityBoundary, isEqual );
  FRIEND_TEST( boundary, pressure );
  FRIEND_TEST( divergenceBoundary, isEqual );
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

#endif //__MACSTABLESOLVER_H
