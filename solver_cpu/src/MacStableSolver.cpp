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
#include <string.h>

#define SWAP(value0,value) { float *tmp=value0;value0=value;value=tmp; }

StableSolverCpu::StableSolverCpu()
{
}

StableSolverCpu::~StableSolverCpu()
{

}

void StableSolverCpu::init()
{
  gridSize.x = 128;
  gridSize.y = 128;

  totCell = gridSize.x * gridSize.y;
  rowVelocity.x = gridSize.x + 1;
  columnVelocity.x = gridSize.y;
  totVelX = rowVelocity.x * columnVelocity.x;
  rowVelocity.y = gridSize.x;
  columnVelocity.y = gridSize.y + 1;
  totVelY = rowVelocity.y*columnVelocity.y;
  min.x = 0.0f;
  max.x = (float)gridSize.x;
  min.y = 0.0f;
  max.y = (float)gridSize.y;

  //params
  running = 1;
  timeStep = 1.0f;
  diffusion = 0.0f;
  viscosity = 0.0f;;

//  velocity.x = (float *)malloc(sizeof(float)*totVelX);
  velocity.x = (float *)malloc(sizeof(float)*totVelX);
  velocity.y = (float *)malloc(sizeof(float)*totVelY);
  previousVelocity.x = (float *)malloc(sizeof(float)*totVelX);
  previousVelocity.y = (float *)malloc(sizeof(float)*totVelY);
  density = (float *)malloc(sizeof(float)*totCell);
  previousDensity = (float *)malloc(sizeof(float)*totCell);
  divergence = (float *)malloc(sizeof(float)*totCell);
  pressure = (float *)malloc(sizeof(float)*totCell);
  pvx = (glm::vec2 *)malloc(sizeof(glm::vec2)*totVelX);
  pvy = (glm::vec2 *)malloc(sizeof(glm::vec2)*totVelY);

  for(int i=0; i<rowVelocity.x; i++)
  {
    for(int j=0; j<columnVelocity.x; j++)
    {
      pvx[vxIdx(i, j)].x = (float)i;
      pvx[vxIdx(i, j)].y = (float)j+0.5f;
      pvy[vyIdx(i, j)].x = (float)i+0.5f;
      pvy[vyIdx(i, j)].y = (float)j;
    }
  }

  for(int i=0; i<rowVelocity.y; i++)
  {
    for(int j=0; j<columnVelocity.y; j++)
    {
      pvy[vyIdx(i, j)].x = (float)i+0.5f;
      pvy[vyIdx(i, j)].y = (float)j;
    }
  }
}

void StableSolverCpu::reset()
{
  for(int i=0; i<totCell; i++) density[i] = 0.0f;
  for(int i=0; i<totVelX; i++) velocity.x[i] = 0.0f;
  for(int i=0; i<totVelY; i++) velocity.y[i] = 0.0f;
}

void StableSolverCpu::cleanBuffer()
{
  for(int i=0; i<totCell; i++) previousDensity[i] = 0.0f;
  for(int i=0; i<totVelX; i++) previousVelocity.x[i] = 0.0f;
  for(int i=0; i<totVelY; i++) previousVelocity.y[i] = 0.0f;
}

void StableSolverCpu::setVelBoundary(int flag)
{
  //x-axis
  if(flag == 1)
  {
    for(int i=1; i<=rowVelocity.x-2; i++)
    {
      velocity.x[vxIdx(i, 0)] = velocity.x[vxIdx(i, 1)];
      velocity.x[vxIdx(i, columnVelocity.x-1)] = velocity.x[vxIdx(i, columnVelocity.x-2)];
    }
    for(int j=1; j<=columnVelocity.x-2; j++)
    {
      velocity.x[vxIdx(0, j)] = -velocity.x[vxIdx(1, j)];
      velocity.x[vxIdx(rowVelocity.x-1, j)] = -velocity.x[vxIdx(rowVelocity.x-2, j)];
    }
    velocity.x[vxIdx(0, 0)] = (velocity.x[vxIdx(1, 0)]+velocity.x[vxIdx(0, 1)])/2;
    velocity.x[vxIdx(rowVelocity.x-1, 0)] = (velocity.x[vxIdx(rowVelocity.x-2, 0)]+velocity.x[vxIdx(rowVelocity.x-1, 1)])/2;
    velocity.x[vxIdx(0, columnVelocity.x-1)] = (velocity.x[vxIdx(1, columnVelocity.x-1)]+velocity.x[vxIdx(0, columnVelocity.x-2)])/2;
    velocity.x[vxIdx(rowVelocity.x-1, columnVelocity.x-1)] = (velocity.x[vxIdx(rowVelocity.x-2, columnVelocity.x-1)]+velocity.x[vxIdx(rowVelocity.x-1, columnVelocity.x-2)])/2;
  }

  //y-axis
  if(flag == 2)
  {
    for(int i=1; i<=rowVelocity.y-2; i++)
    {
      velocity.y[vyIdx(i, 0)] = -velocity.y[vyIdx(i, 1)];
      velocity.y[vyIdx(i, columnVelocity.y-1)] = -velocity.y[vyIdx(i, columnVelocity.y-2)];
    }
    for(int j=1; j<=columnVelocity.y-2; j++)
    {
      velocity.y[vyIdx(0, j)] = velocity.y[vyIdx(1, j)];
      velocity.y[vyIdx(rowVelocity.y-1, j)] = velocity.y[vyIdx(rowVelocity.y-2, j)];
    }
    velocity.y[vyIdx(0, 0)] = (velocity.y[vyIdx(1, 0)]+velocity.y[vyIdx(0, 1)])/2;
    velocity.y[vyIdx(rowVelocity.y-1, 0)] = (velocity.y[vyIdx(rowVelocity.y-2, 0)]+velocity.y[vyIdx(rowVelocity.y-1, 1)])/2;
    velocity.y[vyIdx(0, columnVelocity.y-1)] = (velocity.y[vyIdx(1, columnVelocity.y-1)]+velocity.y[vyIdx(0, columnVelocity.y-2)])/2;
    velocity.y[vyIdx(rowVelocity.y-1, columnVelocity.y-1)] = (velocity.y[vyIdx(rowVelocity.y-2, columnVelocity.y-1)]+velocity.y[vyIdx(rowVelocity.y-1, columnVelocity.y-2)])/2;
  }
}

void StableSolverCpu::setCellBoundary(float *value)
{
  for(int i=1; i<=gridSize.x-2; i++)
  {
    value[cIdx(i, 0)] = value[cIdx(i, 1)];
    value[cIdx(i, gridSize.y-1)] = value[cIdx(i, gridSize.y-2)];
  }
  for(int j=1; j<=gridSize.y-2; j++)
  {
    value[cIdx(0, j)] = value[cIdx(1, j)];
    value[cIdx(gridSize.x-1, j)] = value[cIdx(gridSize.x-2, j)];
  }
  value[cIdx(0, 0)] = (value[cIdx(1, 0)]+value[cIdx(0, 1)])/2;
  value[cIdx(gridSize.x-1, 0)] = (value[cIdx(gridSize.x-2, 0)]+value[cIdx(gridSize.x-1, 1)])/2;
  value[cIdx(0, gridSize.y-1)] = (value[cIdx(1, gridSize.y-1)]+value[cIdx(0, gridSize.y-2)])/2;
  value[cIdx(gridSize.x-1, gridSize.y-1)] = (value[cIdx(gridSize.x-1, gridSize.y-2)]+value[cIdx(gridSize.x-1, gridSize.y-2)])/2;
}

void StableSolverCpu::projection()
{
  int static count=0;
  for(int i=1; i<=gridSize.x-2; i++)
  {
    for(int j=1; j<=gridSize.y-2; j++)
    {
      divergence[cIdx(i, j)] = (velocity.x[vxIdx(i+1, j)]-velocity.x[vxIdx(i, j)]+velocity.y[vyIdx(i, j+1)]-velocity.y[vyIdx(i, j)]);
      pressure[cIdx(i, j)] = 0.0f;
    }
  }
  count++;
  setCellBoundary(pressure);
  setCellBoundary(divergence);

  //projection iteration
  for(int k=0; k<20; k++)
  {
    for(int i=1; i<=gridSize.x-2; i++)
    {
      for(int j=1; j<=gridSize.y-2; j++)
      {
        pressure[cIdx(i, j)] = (pressure[cIdx(i+1, j)]+pressure[cIdx(i-1, j)]+pressure[cIdx(i, j+1)]+pressure[cIdx(i, j-1)]-divergence[cIdx(i, j)])/4.0f;
      }
    }
    setCellBoundary(pressure);
  }

  //velocity minus grad of Pressure
  for(int i=1; i<=rowVelocity.x-2; i++)
  {
    for(int j=1; j<=columnVelocity.x-2; j++)
    {
      velocity.x[vxIdx(i, j)] -= (pressure[cIdx(i, j)]-pressure[cIdx(i-1, j)]);
    }
  }
  for(int i=1; i<=rowVelocity.y-2; i++)
  {
    for(int j=1; j<=columnVelocity.y-2; j++)
    {
      velocity.y[vyIdx(i, j)] -= (pressure[cIdx(i, j)]-pressure[cIdx(i, j-1)]);
    }
  }
  setVelBoundary(1);
  setVelBoundary(2);
}

void StableSolverCpu::advectVel()
{
  for(int i=1; i<=rowVelocity.x-2; i++)
  {
    for(int j=1; j<=columnVelocity.x-2; j++)
    {
      float nvx = previousVelocity.x[vxIdx(i, j)];
      float nvy = (previousVelocity.y[vyIdx(i-1, j)]+previousVelocity.y[vyIdx(i-1, j+1)]+previousVelocity.y[vyIdx(i, j)]+previousVelocity.y[vyIdx(i, j+1)])/4;

      float oldX = pvx[vxIdx(i, j)].x - nvx*timeStep;
      float oldY = pvx[vxIdx(i, j)].y - nvy*timeStep;

      if(oldX < 0.5f) oldX = 0.5f;
      if(oldX > max.x-0.5f) oldX = max.x-0.5f;
      if(oldY < 1.0f) oldY = 1.0f;
      if(oldY > max.y-1.0f) oldY = max.y-1.0f;

      int i0 = (int)oldX;
      int j0 = (int)(oldY-0.5f);
      int i1 = i0+1;
      int j1 = j0+1;

      float wL = pvx[vxIdx(i1, j0)].x-oldX;
      float wR = 1.0f-wL;
      float wB = pvx[vxIdx(i0, j1)].y-oldY;
      float wT = 1.0f-wB;

      //printf("%f, %f, %f, %f\n", wL, wR, wB, wT);

      velocity.x[vxIdx(i, j)] = wB*(wL*previousVelocity.x[vxIdx(i0, j0)]+wR*previousVelocity.x[vxIdx(i1, j0)])+
          wT*(wL*previousVelocity.x[vxIdx(i0, j1)]+wR*previousVelocity.x[vxIdx(i1, j1)]);
    }
  }

  for(int i=1; i<=rowVelocity.y-2; i++)
  {
    for(int j=1; j<=columnVelocity.y-2; j++)
    {
      float nvx = (previousVelocity.x[vxIdx(i, j-1)]+previousVelocity.x[vxIdx(i+1, j-1)]+previousVelocity.x[vxIdx(i, j)]+previousVelocity.x[vxIdx(i+1, j)])/4;
      float nvy = previousVelocity.y[vyIdx(i, j)];

      float oldX = pvy[vyIdx(i, j)].x - nvx*timeStep;
      float oldY = pvy[vyIdx(i, j)].y - nvy*timeStep;

      if(oldX < 1.0f) oldX = 1.0f;
      if(oldX > max.x-1.0f) oldX = max.x-1.0f;
      if(oldY < 0.5f) oldY = 0.5f;
      if(oldY > max.y-0.5f) oldY = max.y-0.5f;

      int i0 = (int)(oldX-0.5f);
      int j0 = (int)oldY;
      int i1 = i0+1;
      int j1 = j0+1;

      float wL = pvy[vyIdx(i1, j0)].x-oldX;
      float wR = 1.0f-wL;
      float wB = pvy[vyIdx(i0, j1)].y-oldY;
      float wT = 1.0f-wB;

      velocity.y[vyIdx(i, j)] = wB*(wL*previousVelocity.y[vyIdx(i0, j0)]+wR*previousVelocity.y[vyIdx(i1, j0)])+
          wT*(wL*previousVelocity.y[vyIdx(i0, j1)]+wR*previousVelocity.y[vyIdx(i1, j1)]);
    }
  }

  setVelBoundary(1);
  setVelBoundary(2);
}

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

  for(int i=1; i<=gridSize.x-2; i++)
  {
    for(int j=1; j<=gridSize.y-2; j++)
    {
      float cvx = getCellVel(i, j).x;
      float cvy = getCellVel(i, j).y;

      oldX = (float)i+0.5f - cvx*timeStep;
      oldY = (float)j+0.5f - cvy*timeStep;

      if(oldX < 1.0f) oldX = 1.0f;
      if(oldX > gridSize.x-1.0f) oldX = gridSize.x-1.0f;
      if(oldY < 1.0f) oldY = 1.0f;
      if(oldY > gridSize.y-1.0f) oldY = gridSize.y-1.0f;

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

  setCellBoundary(density);
}

void StableSolverCpu::diffuseVel()
{
  for(int i=0; i<totVelX; i++) velocity.x[i] = 0.0f;
  for(int i=0; i<totVelY; i++) velocity.y[i] = 0.0f;
  float a = diffusion*timeStep;

  for(int k=0; k<20; k++)
  {
    //diffuse velX
    for(int i=1; i<=rowVelocity.x-2; i++)
    {
      for(int j=1; j<=columnVelocity.x-2; j++)
      {
        velocity.x[vxIdx(i, j)] = (previousVelocity.x[vxIdx(i, j)]+a*(velocity.x[vxIdx(i+1, j)]+velocity.x[vxIdx(i-1, j)]+velocity.x[vxIdx(i, j+1)]+velocity.x[vxIdx(i, j-1)])) / (4.0f*a+1.0f);
      }
    }
    //diffuse velY
    for(int i=1; i<=rowVelocity.y-2; i++)
    {
      for(int j=1; j<=columnVelocity.y-2; j++)
      {
        velocity.y[vyIdx(i, j)] = (previousVelocity.y[vyIdx(i, j)]+a*(velocity.y[vyIdx(i+1, j)]+velocity.y[vyIdx(i-1, j)]+velocity.y[vyIdx(i, j+1)]+velocity.y[vyIdx(i, j-1)])) / (4.0f*a+1.0f);
      }
    }

    //boundary
    setVelBoundary(1);
    setVelBoundary(2);
  }
}

void StableSolverCpu::diffuseCell(float *value, float *value0)
{
  for(int i=0; i<totCell; i++) value[i] = 0.0f;
  float a = viscosity*timeStep;

  for(int k=0; k<20; k++)
  {
    for(int i=1; i<=gridSize.x-2; i++)
    {
      for(int j=1; j<=gridSize.y-2; j++)
      {
        value[cIdx(i, j)] = (value0[cIdx(i, j)]+a*(value[cIdx(i+1, j)]+value[cIdx(i-1, j)]+value[cIdx(i, j+1)]+value[cIdx(i, j-1)])) / (4.0f*a+1.0f);
      }
    }
    setCellBoundary(value);
  }
}

void StableSolverCpu::addSource()
{
  for(int i=0; i<totCell; i++) density[i] += previousDensity[i];
  for(int i=0; i<totVelX; i++) velocity.x[i] += previousVelocity.x[i];
  for(int i=0; i<totVelY; i++) velocity.y[i] += previousVelocity.y[i];

  setVelBoundary(1);
  setVelBoundary(2);
  setCellBoundary(density);
}

void StableSolverCpu::animVel()
{
  projection();

  if(diffusion > 0.0f)
  {
    SWAP(previousVelocity.x, velocity.x);
    SWAP(previousVelocity.y, velocity.y);
    diffuseVel();
  }

  SWAP(previousVelocity.x, velocity.x);
  SWAP(previousVelocity.y, velocity.y);
  advectVel();

  projection();
}

void StableSolverCpu::animDen()
{
  if(viscosity > 0.0f)
  {
    SWAP(previousDensity, density);
    diffuseCell(density, previousDensity);
  }

  SWAP(previousDensity, density);
  advectCell(density, previousDensity);
}
