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

//----------------------------------------------------------------------------------------------------------------------

StableSolverCpu::~StableSolverCpu()
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

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::activate()
{
  m_gridSize.x = 128;
  m_gridSize.y = 128;

  m_totCell = m_gridSize.x * m_gridSize.y;
  m_rowVelocity.x = m_gridSize.x + 1;
  m_rowVelocity.y = m_gridSize.x;

  m_columnVelocity.x = m_gridSize.y;
  m_columnVelocity.y = m_gridSize.y + 1;

  m_totVelX = m_rowVelocity.x * m_columnVelocity.x;
  m_totVelY = m_rowVelocity.y * m_columnVelocity.y;

  m_min.x = 0.0f;
  m_max.x = (float)m_gridSize.x;
  m_min.y = 0.0f;
  m_max.y = (float)m_gridSize.y;

  //params
  m_timeStep = 1.0f;
  m_diffusion = 0.0f;
  m_viscosity = 0.0f;
  m_inputDensity = 100.0f;

  m_velocity.x = (float *)malloc(sizeof(float)*m_totVelX);
  m_velocity.y = (float *)malloc(sizeof(float)*m_totVelY);
  m_previousVelocity.x = (float *)malloc(sizeof(float)*m_totVelX);
  m_previousVelocity.y = (float *)malloc(sizeof(float)*m_totVelY);
  m_density = (float *)malloc(sizeof(float)*m_totCell);
  m_previousDensity = (float *)malloc(sizeof(float)*m_totCell);
  m_divergence = (float *)malloc(sizeof(float)*m_totCell);
  m_pressure = (float *)malloc(sizeof(float)*m_totCell);
  m_pvx = (tuple<float> *)malloc(sizeof(tuple<float>)*m_totVelX);
  m_pvy = (tuple<float> *)malloc(sizeof(tuple<float>)*m_totVelY);

  for(int i=0; i<m_rowVelocity.x; ++i)
  {
    for(int j=0; j<m_columnVelocity.x; ++j)
    {
      m_pvx[vxIdx(i, j)].x = (float)i;
      m_pvx[vxIdx(i, j)].y = (float)j+0.5f;
    }
  }

  for(int i=0; i<m_rowVelocity.y; ++i)
  {
    for(int j=0; j<m_columnVelocity.y; ++j)
    {
      m_pvy[vyIdx(i, j)].x = (float)i+0.5f;
      m_pvy[vyIdx(i, j)].y = (float)j;
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
    for(int i=1; i<=m_rowVelocity.x-2; ++i)
    {
      m_velocity.x[vxIdx(i, 0)] = m_velocity.x[vxIdx(i, 1)];
      m_velocity.x[vxIdx(i, m_columnVelocity.x-1)] = m_velocity.x[vxIdx(i, m_columnVelocity.x-2)];
    }
    for(int j=1; j<=m_columnVelocity.x-2; ++j)
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
    for(int i=1; i<=m_rowVelocity.y-2; ++i)
    {
      m_velocity.y[vyIdx(i, 0)] = -m_velocity.y[vyIdx(i, 1)];
      m_velocity.y[vyIdx(i, m_columnVelocity.y-1)] = -m_velocity.y[vyIdx(i, m_columnVelocity.y-2)];
    }
    for(int j=1; j<=m_columnVelocity.y-2; ++j)
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

void StableSolverCpu::setCellBoundary(float *value)
{
  for(int i=1; i<=m_gridSize.x-2; ++i)
  {
    value[cIdx(i, 0)] = value[cIdx(i, 1)];
    value[cIdx(i, m_gridSize.y-1)] = value[cIdx(i, m_gridSize.y-2)];
  }
  for(int j=1; j<=m_gridSize.y-2; ++j)
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
  int static count=0;
  for(int i=1; i<=m_gridSize.x-2; ++i)
  {
    for(int j=1; j<=m_gridSize.y-2; ++j)
    {
      m_divergence[cIdx(i, j)] = (m_velocity.x[vxIdx(i+1, j)]-m_velocity.x[vxIdx(i, j)]+m_velocity.y[vyIdx(i, j+1)]-m_velocity.y[vyIdx(i, j)]);
      m_pressure[cIdx(i, j)] = 0.0f;
    }
  }
  count++;
  setCellBoundary(m_pressure);
  setCellBoundary(m_divergence);

  //projection iteration
  for(int k=0; k<20; k++)
  {
    for(int i=1; i <= m_gridSize.x-2; ++i)
    {
      for(int j=1; j <= m_gridSize.y-2; ++j)
      {
        m_pressure[cIdx(i, j)] = (m_pressure[cIdx(i+1, j)]+m_pressure[cIdx(i-1, j)]+m_pressure[cIdx(i, j+1)]+m_pressure[cIdx(i, j-1)]-m_divergence[cIdx(i, j)])/4.0f;
      }
    }
    setCellBoundary(m_pressure);
  }

  //velocity minus grad of Pressure
  for(int i=1; i<=m_rowVelocity.x-2; ++i)
  {
    for(int j=1; j<=m_columnVelocity.x-2; ++j)
    {
      m_velocity.x[vxIdx(i, j)] -= (m_pressure[cIdx(i, j)]-m_pressure[cIdx(i-1, j)]);
    }
  }
  for(int i=1; i<=m_rowVelocity.y-2; ++i)
  {
    for(int j=1; j<=m_columnVelocity.y-2; ++j)
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
  for(int i=1; i<=m_rowVelocity.x-2; ++i)
  {
    for(int j=1; j<=m_columnVelocity.x-2; ++j)
    {
      float nvx = m_previousVelocity.x[vxIdx(i, j)];
      float nvy = (m_previousVelocity.y[vyIdx(i-1, j)]+m_previousVelocity.y[vyIdx(i-1, j+1)]+m_previousVelocity.y[vyIdx(i, j)]+m_previousVelocity.y[vyIdx(i, j+1)])/4;

      float oldX = m_pvx[vxIdx(i, j)].x - nvx*m_timeStep;
      float oldY = m_pvx[vxIdx(i, j)].y - nvy*m_timeStep;

      if(oldX < 0.5f) oldX = 0.5f;
      if(oldX > m_max.x-0.5f) oldX = m_max.x-0.5f;
      if(oldY < 1.0f) oldY = 1.0f;
      if(oldY > m_max.y-1.0f) oldY = m_max.y-1.0f;

      int i0 = (int)oldX;
      int j0 = (int)(oldY-0.5f);
      int i1 = i0+1;
      int j1 = j0+1;

      float wL = m_pvx[vxIdx(i1, j0)].x-oldX;
      float wR = 1.0f-wL;
      float wB = m_pvx[vxIdx(i0, j1)].y-oldY;
      float wT = 1.0f-wB;

      //printf("%f, %f, %f, %f\n", wL, wR, wB, wT);

      m_velocity.x[vxIdx(i, j)] = wB*(wL*m_previousVelocity.x[vxIdx(i0, j0)]+wR*m_previousVelocity.x[vxIdx(i1, j0)])+
          wT*(wL*m_previousVelocity.x[vxIdx(i0, j1)]+wR*m_previousVelocity.x[vxIdx(i1, j1)]);
    }
  }

  for(int i=1; i<=m_rowVelocity.y-2; ++i)
  {
    for(int j=1; j<=m_columnVelocity.y-2; ++j)
    {
      float nvx = (m_previousVelocity.x[vxIdx(i, j-1)]+m_previousVelocity.x[vxIdx(i+1, j-1)]+m_previousVelocity.x[vxIdx(i, j)]+m_previousVelocity.x[vxIdx(i+1, j)])/4;
      float nvy = m_previousVelocity.y[vyIdx(i, j)];

      float oldX = m_pvy[vyIdx(i, j)].x - nvx*m_timeStep;
      float oldY = m_pvy[vyIdx(i, j)].y - nvy*m_timeStep;

      if(oldX < 1.0f) oldX = 1.0f;
      if(oldX > m_max.x-1.0f) oldX = m_max.x-1.0f;
      if(oldY < 0.5f) oldY = 0.5f;
      if(oldY > m_max.y-0.5f) oldY = m_max.y-0.5f;

      int i0 = (int)(oldX-0.5f);
      int j0 = (int)oldY;
      int i1 = i0+1;
      int j1 = j0+1;

      float wL = m_pvy[vyIdx(i1, j0)].x-oldX;
      float wR = 1.0f-wL;
      float wB = m_pvy[vyIdx(i0, j1)].y-oldY;
      float wT = 1.0f-wB;

      m_velocity.y[vyIdx(i, j)] = wB*(wL*m_previousVelocity.y[vyIdx(i0, j0)]+wR*m_previousVelocity.y[vyIdx(i1, j0)])+
          wT*(wL*m_previousVelocity.y[vyIdx(i0, j1)]+wR*m_previousVelocity.y[vyIdx(i1, j1)]);
    }
  }

  setVelBoundary(1);
  setVelBoundary(2);
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::advectCell(float *value, float *value0)
{
  float oldX;
  float oldY;
  int i0;
  int i1;
  int j0;
  int j1;
  float wL;
  float wR;
  float wB;
  float wT;

  for(int i=1; i<=m_gridSize.x-2; ++i)
  {
    for(int j=1; j<=m_gridSize.y-2; ++j)
    {
      float cvx = getCellVel(i, j).x;
      float cvy = getCellVel(i, j).y;

      oldX = (float)i+0.5f - cvx*m_timeStep;
      oldY = (float)j+0.5f - cvy*m_timeStep;

      if(oldX < 1.0f) oldX = 1.0f;
      if(oldX > m_gridSize.x-1.0f) oldX = m_gridSize.x-1.0f;
      if(oldY < 1.0f) oldY = 1.0f;
      if(oldY > m_gridSize.y-1.0f) oldY = m_gridSize.y-1.0f;

      i0 = (int)(oldX-0.5f);
      j0 = (int)(oldY-0.5f);
      i1 = i0+1;
      j1 = j0+1;

      wL = (float)i1+0.5f-oldX;
      wR = 1.0f-wL;
      wB = (float)j1+0.5f-oldY;
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
  for(int i=0; i<m_totVelX; ++i) m_velocity.x[i] = 0.0f;
  for(int i=0; i<m_totVelY; ++i) m_velocity.y[i] = 0.0f;
  float a = m_diffusion*m_timeStep;

  for(int k=0; k<20; k++)
  {
    //diffuse velX
    for(int i=1; i<=m_rowVelocity.x-2; ++i)
    {
      for(int j=1; j<=m_columnVelocity.x-2; ++j)
      {
        m_velocity.x[vxIdx(i, j)] = (m_previousVelocity.x[vxIdx(i, j)]+a*(m_velocity.x[vxIdx(i+1, j)]+m_velocity.x[vxIdx(i-1, j)]+m_velocity.x[vxIdx(i, j+1)]+m_velocity.x[vxIdx(i, j-1)])) / (4.0f*a+1.0f);
      }
    }
    //diffuse velY
    for(int i=1; i<=m_rowVelocity.y-2; ++i)
    {
      for(int j=1; j<=m_columnVelocity.y-2; ++j)
      {
        m_velocity.y[vyIdx(i, j)] = (m_previousVelocity.y[vyIdx(i, j)]+a*(m_velocity.y[vyIdx(i+1, j)]+m_velocity.y[vyIdx(i-1, j)]+m_velocity.y[vyIdx(i, j+1)]+m_velocity.y[vyIdx(i, j-1)])) / (4.0f*a+1.0f);
      }
    }

    //boundary
    setVelBoundary(1);
    setVelBoundary(2);
  }
}

//----------------------------------------------------------------------------------------------------------------------

void StableSolverCpu::diffuseCell(float *value, float *value0)
{
  for(int i=0; i<m_totCell; ++i) value[i] = 0.0f;
  float a = m_viscosity*m_timeStep;

  for(int k=0; k<20; ++k)
  {
    for(int i=1; i<=m_gridSize.x-2; ++i)
    {
      for(int j=1; j<=m_gridSize.y-2; ++j)
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

  if(m_diffusion > 0.0f)
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
void StableSolverCpu::exportCSV( std::string _file, tuple<float> * _t, int _sizeX, int _sizeY )
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
