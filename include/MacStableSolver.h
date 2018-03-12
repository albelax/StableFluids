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

#include <glm.hpp>
#include <stdio.h>


template <class T>
class vec2
{
public:
    vec2() = default;
    T x;
    T y;
};

class StableSolverCpu
{
public:
    StableSolverCpu();
    ~StableSolverCpu();
    void init();
    void reset();
    void cleanBuffer();
    void start(){ running=1; }
    void stop(){ running=0; }
    int isRunning(){ return running; }

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
    int getRowCell(){ return gridSize.x; }
    int getColCell(){ return gridSize.y; }
    int getTotCell(){ return totCell; }
    int getRowVelX(){ return rowVelocity.x; }
    int getcolVelX(){ return columnVelocity.x; }
    int getTotVelX(){ return totVelX; }
    int getRowVelY(){ return rowVelocity.y; }
    int getColVelY(){ return columnVelocity.y; }
    int getTotVelY(){ return totVelY; }
    float* getVX(){ return velocity.x; }
    float* getVY(){ return velocity.y; }
    float* getD(){ return density; }
    int vxIdx(int i, int j){ return j*rowVelocity.x+i; }
    int vyIdx(int i, int j){ return j*rowVelocity.y+i; }
    int cIdx(int i, int j){ return j*gridSize.x+i; }
    glm::vec2* getPVX(){ return pvx; }
    glm::vec2* getPVY(){ return pvy; }
    glm::vec2 getCellVel(int i, int j){ return glm::vec2((velocity.x[vxIdx(i, j)]+velocity.x[vxIdx(i+1, j)])/2, (velocity.y[vyIdx(i, j)]+velocity.y[vyIdx(i, j+1)])/2); }
    
    float getDens(int i, int j) // calculates density of cell
    { 
        return (
        density[cIdx(i-1, j-1)] +
        density[cIdx(i, j-1)] +
        density[cIdx(i-1, j)] +
        density[cIdx(i, j)])/4.0f;
}

    //setter
    void setVel0(int i, int j, float _vx0, float _vy0)
    { 
        previousVelocity.x[vxIdx(i, j)] += _vx0;
        previousVelocity.x[vxIdx(i+1, j)] += _vx0;
        previousVelocity.y[vyIdx(i, j)] += _vy0;
        previousVelocity.y[vyIdx(i, j+1)] += _vy0;
    }
    void setD0(int i, int j, float _d0){ previousDensity[cIdx(i, j)]=_d0; }
private:
    bool running;
    int totCell;
    int totVelX;
    int totVelY;
    float timeStep;
    float diffusion;
    float viscosity;

    float *density;
    float *previousDensity;
    float *divergence;
    float *pressure;
    vec2<int> gridSize;
    vec2<int> rowVelocity;
    vec2<int> columnVelocity;
    vec2<float> min;
    vec2<float> max;
    vec2<float *> velocity;
    vec2<float *> previousVelocity;
    glm::vec2 * pvx;
    glm::vec2 * pvy;
};




#endif
