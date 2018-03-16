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
#include "testing.h"

#define SWAP(value0,value) { float *tmp = value0; value0 = value; value = tmp; }
const int i = 0;
#define TESTING 1

#if TESTING
#include <gtest/gtest.h>
#endif // TESTING

class StableSolverCpu
{
public:
  StableSolverCpu();
  ~StableSolverCpu();
  void init();
  void reset();
  void cleanBuffer();

  //animation
  void setVelBoundary(int flag);
  void setCellBoundary(float *value);
  void projection();
  void advectVel();
  void advectCell(float *value, float *value0);
  void diffuseVel(); // diffuse velocity
  void diffuseCell(float *value, float *value0); // diffuse Density
  void addSource();
  void animVel();
  void animDen();

  //getter
  int getRowCell(){ return m_gridSize.x; }
  int getColCell(){ return m_gridSize.y; }
  int getTotCell(){ return m_totCell; }
  int getRowVelX(){ return m_rowVelocity.x; }
  int getcolVelX(){ return m_columnVelocity.x; }
  int getTotVelX(){ return m_totVelX; }
  int getRowVelY(){ return m_rowVelocity.y; }
  int getColVelY(){ return m_columnVelocity.y; }
  int getTotVelY(){ return m_totVelY; }
  int vxIdx(int i, int j){ return j*m_rowVelocity.x+i; }
  int vyIdx(int i, int j){ return j*m_rowVelocity.y+i; }
  int cIdx(int i, int j){ return j*m_gridSize.x+i; }
  float* getVX(){ return m_velocity.x; }
  float* getVY(){ return m_velocity.y; }
  float* getD(){ return m_density; }
  tuple<float> * getPVX(){ return m_pvx; }
  tuple<float> * getPVY(){ return m_pvy; }
  tuple<float> getCellVel(int i, int j)
  {
    float x = (m_velocity.x[vxIdx(i, j)]+m_velocity.x[vxIdx(i+1, j)]) / 2.0f;
    float y = (m_velocity.y[vyIdx(i, j)]+m_velocity.y[vyIdx(i, j+1)]) / 2.0f;
    tuple<float> ret( x, y );
    return ret;
  }

  float getDens(int i, int j) // calculates density of cell
  {
    return (
          m_density[cIdx(i-1, j-1)] +
        m_density[cIdx(i, j-1)] +
        m_density[cIdx(i-1, j)] +
        m_density[cIdx(i, j)]) / 4.0f;
  }

  void setVel0(int i, int j, float _vx0, float _vy0)
  {
    m_previousVelocity.x[vxIdx(i, j)] += _vx0;
    m_previousVelocity.x[vxIdx(i+1, j)] += _vx0;
    m_previousVelocity.y[vyIdx(i, j)] += _vy0;
    m_previousVelocity.y[vyIdx(i, j+1)] += _vy0;
  }
  void setD0(int i, int j, float _d0){ m_previousDensity[cIdx(i, j)]=_d0; }
  void exportCSV( std::string _file );
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

#if TESTING
  FRIEND_TEST(pvx, isEqual);
#endif // TESTING
};




#endif
