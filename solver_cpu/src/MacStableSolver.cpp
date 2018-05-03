/** File:    MacStableSolver.cpp
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

#include "MacStableSolver.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>
#include <time.h>
#include "parameters.h"


StableSolverCpu::StableSolverCpu()
{

}
//----------------------------------------------------------------------------------------------------------------------

StableSolverCpu::~StableSolverCpu()
{
  if ( m_active )
  {
    free( m_pvx );
    free( m_pvy );
    free( m_density );
    free( m_pressure );
    free( m_divergence );
    free( m_velocity.x );
    free( m_velocity.y );
    free( m_previousVelocity.x );
    free( m_previousVelocity.y );
    free( m_previousDensity );
  }
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::activate()
{
  m_active = true;
  m_gridSize.x = Common::gridWidth;
  m_gridSize.y = Common::gridHeight;
  m_totCell = Common::totCells;
  m_rowVelocity.x = Common::rowVelocityX;
  m_rowVelocity.y = Common::rowVelocityY;

  m_columnVelocity.x = Common::columnVelocityX;
  m_columnVelocity.y = Common::columnVelocityY;

  m_totVelX = Common::totHorizontalVelocity;
  m_totVelY = Common::totVerticalVelocity;

  m_min.x = 0.0f;
  m_max.x = (real) m_gridSize.x;
  m_min.y = 0.0f;
  m_max.y = (real) m_gridSize.y;

  //params
  m_timeStep = 1.0f;
  m_diffusion = 0.0f;
  m_viscosity = 0.0f;
  m_inputDensity = 100.0f;

  m_velocity.x = (real *)malloc(sizeof(real)*m_totVelX);
  m_velocity.y = (real *)malloc(sizeof(real)*m_totVelY);
  m_previousVelocity.x = (real *)malloc(sizeof(real)*m_totVelX);
  m_previousVelocity.y = (real *)malloc(sizeof(real)*m_totVelY);
  m_density = (real *)malloc(sizeof(real)*m_totCell);
  m_previousDensity = (real *)malloc(sizeof(real)*m_totCell);
  m_divergence = (real *)malloc(sizeof(real)*m_totCell);
  m_pressure = (real *)malloc(sizeof(real)*m_totCell);
  m_pvx = (tuple<real> *)malloc(sizeof(tuple<real>)*m_totVelX);
  m_pvy = (tuple<real> *)malloc(sizeof(tuple<real>)*m_totVelY);

  for(unsigned int i = 0; i < m_rowVelocity.x; ++i)
  {
    for(unsigned int j = 0; j < m_columnVelocity.x; ++j)
    {
      m_pvx[vxIdx(i, j)].x = (real)i;
      m_pvx[vxIdx(i, j)].y = (real)j+0.5f;
    }
  }

  for(unsigned int i = 0; i < m_rowVelocity.y; ++i )
  {
    for(unsigned int j = 0; j < m_columnVelocity.y; ++j )
    {
      m_pvy[vyIdx(i, j)].x = (real)i+0.5f;
      m_pvy[vyIdx(i, j)].y = (real)j;
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::reset()
{
  for(int i=0; i<m_totCell; ++i) m_density[i] = 0.0f;
  for(int i=0; i<m_totVelX; ++i) m_velocity.x[i] = 0.0f;
  for(int i=0; i<m_totVelY; ++i) m_velocity.y[i] = 0.0f;
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::cleanBuffer()
{
  for(int i=0; i<m_totCell; ++i) m_previousDensity[i] = 0.0f;
  for(int i=0; i<m_totVelX; ++i) m_previousVelocity.x[i] = 0.0f;
  for(int i=0; i<m_totVelY; ++i) m_previousVelocity.y[i] = 0.0f;
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::setVelBoundary(int flag)
{
  //x-axis
  if(flag == 1)
  {
    for(unsigned int i=1; i<=m_rowVelocity.x-2; ++i)
    {
      m_velocity.x[vxIdx(i, 0)] = m_velocity.x[vxIdx(i, 1)];
      m_velocity.x[vxIdx(i, m_columnVelocity.x-1)] = m_velocity.x[vxIdx(i, m_columnVelocity.x-2)];
    }
    for(unsigned int j=1; j<=m_columnVelocity.x-2; ++j)
    {
      m_velocity.x[vxIdx(0, j)] = -m_velocity.x[vxIdx(1, j)];
      m_velocity.x[vxIdx(m_rowVelocity.x-1, j)] = -m_velocity.x[vxIdx(m_rowVelocity.x-2, j)];
    }
    m_velocity.x[vxIdx(0, 0)] = (m_velocity.x[vxIdx(1, 0)]+m_velocity.x[vxIdx(0, 1)])/2;
    m_velocity.x[vxIdx(m_rowVelocity.x-1, 0)] = (m_velocity.x[vxIdx(m_rowVelocity.x-2, 0)]+m_velocity.x[vxIdx(m_rowVelocity.x-1, 1)])/2;
    m_velocity.x[vxIdx(0, m_columnVelocity.x-1)] = (m_velocity.x[vxIdx(1, m_columnVelocity.x-1)]+m_velocity.x[vxIdx(0, m_columnVelocity.x-2)])/2;
    m_velocity.x[vxIdx(m_rowVelocity.x-1, m_columnVelocity.x-1)] = (m_velocity.x[vxIdx(m_rowVelocity.x-2, m_columnVelocity.x-1)]+m_velocity.x[vxIdx(m_rowVelocity.x-1, m_columnVelocity.x-2)])/2;
  }

  //y-axis
  if(flag == 2)
  {
    for(unsigned int i=1; i<=m_rowVelocity.y-2; ++i)
    {
      m_velocity.y[vyIdx(i, 0)] = -m_velocity.y[vyIdx(i, 1)];
      m_velocity.y[vyIdx(i, m_columnVelocity.y-1)] = -m_velocity.y[vyIdx(i, m_columnVelocity.y-2)];
    }
    for(unsigned int j=1; j<=m_columnVelocity.y-2; ++j)
    {
      m_velocity.y[vyIdx(0, j)] = m_velocity.y[vyIdx(1, j)];
      m_velocity.y[vyIdx(m_rowVelocity.y-1, j)] = m_velocity.y[vyIdx(m_rowVelocity.y-2, j)];
    }
    m_velocity.y[vyIdx(0, 0)] = (m_velocity.y[vyIdx(1, 0)]+m_velocity.y[vyIdx(0, 1)])/2;
    m_velocity.y[vyIdx(m_rowVelocity.y-1, 0)] = (m_velocity.y[vyIdx(m_rowVelocity.y-2, 0)]+m_velocity.y[vyIdx(m_rowVelocity.y-1, 1)])/2;
    m_velocity.y[vyIdx(0, m_columnVelocity.y-1)] = (m_velocity.y[vyIdx(1, m_columnVelocity.y-1)]+m_velocity.y[vyIdx(0, m_columnVelocity.y-2)])/2;
    m_velocity.y[vyIdx(m_rowVelocity.y-1, m_columnVelocity.y-1)] = (m_velocity.y[vyIdx(m_rowVelocity.y-2, m_columnVelocity.y-1)]+m_velocity.y[vyIdx(m_rowVelocity.y-1, m_columnVelocity.y-2)])/2;
  }
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::setCellBoundary(real *value)
{
  for(unsigned int i=1; i<=m_gridSize.x-2; ++i)
  {
    value[cIdx(i, 0)] = value[cIdx(i, 1)];
    value[cIdx(i, m_gridSize.y-1)] = value[cIdx(i, m_gridSize.y-2)];
  }
  for(unsigned int j=1; j<=m_gridSize.y-2; ++j)
  {
    value[cIdx(0, j)] = value[cIdx(1, j)];
    value[cIdx(m_gridSize.x-1, j)] = value[cIdx(m_gridSize.x-2, j)];
  }
  value[cIdx(0, 0)] = (value[cIdx(1, 0)]+value[cIdx(0, 1)])/2;
  value[cIdx(m_gridSize.x-1, 0)] = (value[cIdx(m_gridSize.x-2, 0)]+value[cIdx(m_gridSize.x-1, 1)])/2;
  value[cIdx(0, m_gridSize.y-1)] = (value[cIdx(1, m_gridSize.y-1)]+value[cIdx(0, m_gridSize.y-2)])/2;
  value[cIdx(m_gridSize.x-1, m_gridSize.y-1)] = (value[cIdx(m_gridSize.x-1, m_gridSize.y-2)]+value[cIdx(m_gridSize.x-1, m_gridSize.y-2)])/2;
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::projection()
{
  //  int static count = 0;
  unsigned int iterations = 20;
  for(unsigned int i=1; i<=m_gridSize.x-2; ++i)
  {
    for(unsigned int j=1; j<=m_gridSize.y-2; ++j)
    {
      m_divergence[cIdx(i, j)] = (m_velocity.x[vxIdx(i+1, j)]-m_velocity.x[vxIdx(i, j)]+m_velocity.y[vyIdx(i, j+1)]-m_velocity.y[vyIdx(i, j)]);
      m_pressure[cIdx(i, j)] = 0.0;
    }
  }
  //  count++;
  setCellBoundary(m_pressure);
  setCellBoundary(m_divergence);

  //projection iteration
  for(unsigned int k = 0; k < iterations; k++)
  {
    for(unsigned int i=1; i <= m_gridSize.x-2; ++i)
    {
      for(unsigned int j=1; j <= m_gridSize.y-2; ++j)
      {
        m_pressure[cIdx(i, j)] = (m_pressure[cIdx(i+1, j)]
            +m_pressure[cIdx(i-1, j)]
            +m_pressure[cIdx(i, j+1)]+
            m_pressure[cIdx(i, j-1)]
            -m_divergence[cIdx(i, j)])/4.0;
      }
    }
    setCellBoundary(m_pressure);
  }

  //velocity minus grad of Pressure
  for(unsigned int i=1; i<=m_rowVelocity.x-2; ++i)
  {
    for(unsigned int j=1; j<=m_columnVelocity.x-2; ++j)
    {
      m_velocity.x[vxIdx(i, j)] -= (m_pressure[cIdx(i, j)] -m_pressure[cIdx(i-1, j)]);
    }
  }

  for(unsigned int i=1; i<=m_rowVelocity.y-2; ++i)
  {
    for(unsigned int j=1; j<=m_columnVelocity.y-2; ++j)
    {
      m_velocity.y[vyIdx(i, j)] -= (m_pressure[cIdx(i, j)]-m_pressure[cIdx(i, j-1)]);
    }
  }
  setVelBoundary(1);
  setVelBoundary(2);
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::advectVel()
{
  for(unsigned int i=1; i<=m_rowVelocity.x-2; ++i)
  {
    for(unsigned int j=1; j<=m_columnVelocity.x-2; ++j)
    {
      real nvx = m_previousVelocity.x[vxIdx(i, j)];
      real nvy = (m_previousVelocity.y[vyIdx(i-1, j)]+m_previousVelocity.y[vyIdx(i-1, j+1)]+m_previousVelocity.y[vyIdx(i, j)]+m_previousVelocity.y[vyIdx(i, j+1)])/4;

      real oldX = m_pvx[vxIdx(i, j)].x - nvx*m_timeStep;
      real oldY = m_pvx[vxIdx(i, j)].y - nvy*m_timeStep;

      if(oldX < 0.5f) oldX = 0.5f;
      if(oldX > m_max.x-0.5f) oldX = m_max.x-0.5f;
      if(oldY < 1.0f) oldY = 1.0f;
      if(oldY > m_max.y-1.0f) oldY = m_max.y-1.0f;

      int i0 = (int)oldX;
      int j0 = (int)(oldY-0.5f);
      int i1 = i0+1;
      int j1 = j0+1;

      real wL = m_pvx[vxIdx(i1, j0)].x-oldX;
      real wR = 1.0f-wL;
      real wB = m_pvx[vxIdx(i0, j1)].y-oldY;
      real wT = 1.0f-wB;

      //printf("%f, %f, %f, %f\n", wL, wR, wB, wT);

      m_velocity.x[vxIdx(i, j)] = wB*(wL*m_previousVelocity.x[vxIdx(i0, j0)]+
          wR*m_previousVelocity.x[vxIdx(i1, j0)])+
          wT*(wL*m_previousVelocity.x[vxIdx(i0, j1)]+
          wR*m_previousVelocity.x[vxIdx(i1, j1)]);

    }
  }

  for(unsigned int i=1; i<=m_rowVelocity.y-2; ++i)
  {
    for(unsigned int j=1; j<=m_columnVelocity.y-2; ++j)
    {
      real nvx = (
            m_previousVelocity.x[vxIdx(i, j-1)]+
          m_previousVelocity.x[vxIdx(i+1, j-1)]+
          m_previousVelocity.x[vxIdx(i, j)]+
          m_previousVelocity.x[vxIdx(i+1, j)]
          )/4;

      real nvy = m_previousVelocity.y[vyIdx(i, j)];

      real oldX = m_pvy[vyIdx(i, j)].x - nvx*m_timeStep;
      real oldY = m_pvy[vyIdx(i, j)].y - nvy*m_timeStep;

      if(oldX < 1.0f) oldX = 1.0f;
      if(oldX > m_max.x-1.0f) oldX = m_max.x-1.0f;
      if(oldY < 0.5f) oldY = 0.5f;
      if(oldY > m_max.y-0.5f) oldY = m_max.y-0.5f;

      int i0 = (int)(oldX-0.5f);
      int j0 = (int)oldY;
      int i1 = i0+1;
      int j1 = j0+1;

      real wL = m_pvy[vyIdx(i1, j0)].x-oldX;
      real wR = 1.0f-wL;
      real wB = m_pvy[vyIdx(i0, j1)].y-oldY;
      real wT = 1.0f-wB;

      m_velocity.y[vyIdx(i, j)] = wB*(wL*m_previousVelocity.y[vyIdx(i0, j0)]+
          wR*m_previousVelocity.y[vyIdx(i1, j0)])+
          wT*(wL*m_previousVelocity.y[vyIdx(i0, j1)]+
          wR*m_previousVelocity.y[vyIdx(i1, j1)]);
    }
  }

  setVelBoundary(1);
  setVelBoundary(2);
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::advectCell(real *value, real *value0)
{
  real oldX;
  real oldY;
  int i0;
  int i1;
  int j0;
  int j1;
  real wL;
  real wR;
  real wB;
  real wT;

  for(unsigned int i=1; i<=m_gridSize.x-2; ++i)
  {
    for(unsigned int j=1; j<=m_gridSize.y-2; ++j)
    {
      real cvx = getCellVel(i, j).x;
      real cvy = getCellVel(i, j).y;

      oldX = (real)i+0.5f - cvx*m_timeStep;
      oldY = (real)j+0.5f - cvy*m_timeStep;

      if(oldX < 1.0f) oldX = 1.0f;
      if(oldX > m_gridSize.x-1.0f) oldX = m_gridSize.x-1.0f;
      if(oldY < 1.0f) oldY = 1.0f;
      if(oldY > m_gridSize.y-1.0f) oldY = m_gridSize.y-1.0f;

      i0 = (int)(oldX-0.5f);
      j0 = (int)(oldY-0.5f);
      i1 = i0+1;
      j1 = j0+1;

      wL = (real)i1+0.5f-oldX;
      wR = 1.0f-wL;
      wB = (real)j1+0.5f-oldY;
      wT = 1.0f-wB;

      value[cIdx(i, j)] = wB*(wL*value0[cIdx(i0, j0)]+wR*value0[cIdx(i1, j0)])+
          wT*(wL*value0[cIdx(i0, j1)]+wR*value0[cIdx(i1, j1)]);
    }
  }

  setCellBoundary(m_density);
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::diffuseVel()
{
  for(int i=0; i<m_totVelX; ++i) m_velocity.x[i] = 0;
  for(int i=0; i<m_totVelY; ++i) m_velocity.y[i] = 0;
  real a = m_diffusion * m_timeStep;

  for(int k=0; k<20; k++)
  {
    //diffuse velX
    for(unsigned int i=1; i <= m_rowVelocity.x-2; ++i)
    {
      for(unsigned int j=1; j <= m_columnVelocity.x-2; ++j)
      {
        m_velocity.x[vxIdx(i, j)] = (m_previousVelocity.x[vxIdx(i, j)]+a*(m_velocity.x[vxIdx(i+1, j)]+m_velocity.x[vxIdx(i-1, j)]+m_velocity.x[vxIdx(i, j+1)]+m_velocity.x[vxIdx(i, j-1)])) / (4.0f*a+1.0f);
      }
    }
    //diffuse velY
    for(unsigned int i = 1; i <= m_rowVelocity.y-2; ++i)
    {
      for(unsigned int j = 1; j <= m_columnVelocity.y-2; ++j)
      {
        m_velocity.y[vyIdx(i, j)] = (m_previousVelocity.y[vyIdx(i, j)]+
            a*(m_velocity.y[vyIdx(i+1, j)]+m_velocity.y[vyIdx(i-1, j)]+
            m_velocity.y[vyIdx(i, j+1)]+m_velocity.y[vyIdx(i, j-1)])) / (4.0f*a+1.0f);
      }
    }

    //boundary
    setVelBoundary(1);
    setVelBoundary(2);
  }
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::diffuseCell(real *value, real *value0)
{
  for(int i=0; i<m_totCell; ++i) value[i] = 0;
  real a = m_viscosity*m_timeStep;

  for( int k = 0; k < 20; ++k )
  {
    for(unsigned int i = 1; i <= m_gridSize.x - 2; ++i )
    {
      for(unsigned int j = 1; j <= m_gridSize.y - 2; ++j )
      {
        value[cIdx(i, j)] = (value0[cIdx(i, j)]+a*(value[cIdx(i+1, j)]+value[cIdx(i-1, j)]+value[cIdx(i, j+1)]+value[cIdx(i, j-1)])) / (4.0f*a+1.0f);
      }
    }
    setCellBoundary(value);
  }
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::addSource()
{
  for(int i=0; i<m_totCell; ++i) m_density[i] += m_previousDensity[i];
  for(int i=0; i<m_totVelX; ++i) m_velocity.x[i] += m_previousVelocity.x[i];
  for(int i=0; i<m_totVelY; ++i) m_velocity.y[i] += m_previousVelocity.y[i];

  setVelBoundary(1);
  setVelBoundary(2);
  setCellBoundary( m_density );
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::animVel()
{
  projection();

  if( m_diffusion > 0.0f )
  {
    SWAP(m_previousVelocity.x, m_velocity.x);
    SWAP(m_previousVelocity.y, m_velocity.y);
    diffuseVel();
  }

  SWAP(m_previousVelocity.x, m_velocity.x);
  SWAP(m_previousVelocity.y, m_velocity.y);
  advectVel();

  projection();
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::animDen()
{
  if(m_viscosity > 0.0f)
  {
    SWAP(m_previousDensity, m_density);
    diffuseCell(m_density, m_previousDensity);
  }

  SWAP(m_previousDensity, m_density);
  advectCell(m_density, m_previousDensity);
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::exportCSV( std::string _file, tuple<real> * _t, int _sizeX, int _sizeY )
{
  std::ofstream out;
  out.open( _file );
  out.clear();

  for(int i = 0; i < _sizeX; ++i)
  {
    for(int j = 0; j < _sizeY; ++j)
    {
      int idx = j * _sizeX + i;
      out << "( " << _t[idx].x << ", " << _t[idx].y << " )" << "; ";
    }
    out << "\n";
  }
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::randomizeArrays()
{
  // really bad,
  // used only for testing and allocate data before a benchmark

  //----------- cell based arrays
  std::vector<real> tmp_cell;
  tmp_cell.reserve( m_totCell );
  Rand_CPU::randFloats( tmp_cell );
  for ( int i = 0; i < m_totCell; ++i )
  {
    m_pressure[i] = tmp_cell[i];
    m_divergence[i] = tmp_cell[i];
  }
  //----------- X VELOCITY
  std::vector<real> tmp_vel_x;
  tmp_vel_x.reserve( m_totVelX );
  Rand_CPU::randFloats( tmp_vel_x );
  for ( int i = 0; i < m_totVelX; ++i )
  {
    m_velocity.x[i] = tmp_vel_x[i];
  }
  //----------- Y VELOCITY
  std::vector<real> tmp_vel_y;
  tmp_vel_y.reserve( m_totVelY );
  Rand_CPU::randFloats( tmp_vel_y );
  for ( int i = 0; i < m_totVelX; ++i )
  {
    m_velocity.y[i] = tmp_vel_y[i];
  }
  //-----------

}

//----------------------------------------------------------------------------------------------------------------------
