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
#include "tuple.h"
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
  void activate() override;
  void reset() override;
  void cleanBuffer() override;

  //animation
  void setVelBoundary(int flag);
  void setCellBoundary(real *value);
  void projection();
  void advectVel();
  void advectCell(real *value, real *value0);
  void diffuseVel(); // diffuse velocity
  void diffuseCell(real *value, real *value0); // diffuse Density
  void addSource() override;
  void animVel() override;
  void animDen() override;
  QImage draw( const QImage & _image ) const;

  real* getVX() const { return m_velocity.x; }
  real* getVY() const { return m_velocity.y; }
  real* getD() const { return m_density; }
  tuple<real> * getPVX() const { return m_pvx; }
  tuple<real> * getPVY() const { return m_pvy; }
  const real * getDens() override { return m_density; }

  tuple<real> getCellVel(int i, int j) const
  {
    real x = (m_velocity.x[vxIdx(i, j)]+m_velocity.x[vxIdx(i+1, j)]) / 2.0f;
    real y = (m_velocity.y[vyIdx(i, j)]+m_velocity.y[vyIdx(i, j+1)]) / 2.0f;
    tuple<real> ret( x, y );
    return ret;
  }

  real getDens(int i, int j) const // calculates density of cell
  {
    real dens = m_density[cIdx(i-1, j-1)] + m_density[cIdx(i, j-1)] +
        m_density[cIdx(i-1, j)] + m_density[cIdx(i, j)];
    return dens / 4.0f;
  }

  void setVel0(int i, int j, real _vx0, real _vy0)
  {
    m_previousVelocity.x[vxIdx(i, j)] = _vx0;
    m_previousVelocity.x[vxIdx(i+1, j)] = _vx0;
    m_previousVelocity.y[vyIdx(i, j)] = _vy0;
    m_previousVelocity.y[vyIdx(i, j+1)] = _vy0;
  }
  void setD0(int i, int j ){ m_previousDensity[cIdx(i, j)] = m_inputDensity; }
  void exportCSV( std::string _file, tuple<real> * _t, int _sizeX, int _sizeY );
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
