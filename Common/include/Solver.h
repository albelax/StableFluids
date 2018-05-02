#ifndef SOLVER_H
#define SOLVER_H

#include "tuple.h"

#define SWAP(value0,value) { real *tmp = value0; value0 = value; value = tmp; }

class Solver
{
public:
  Solver();
  ~Solver();
  virtual void cleanBuffer() = 0;
  virtual void activate() = 0;
  virtual void setVel0(int i, int j, real _vx0, real _vy0) = 0;
  virtual void setD0(int i, int j ) = 0;
  virtual void addSource() = 0;
  virtual void animVel() = 0;
  virtual void animDen() = 0;
  virtual void reset() = 0;
  virtual const real * getDens() = 0;

  void setTimestep( real _timeStep ) { m_timeStep = _timeStep; }
  void setDiffusion( real _diffusion ) { m_diffusion = _diffusion; }
  void setViscosity( real _viscosity ) { m_viscosity = _viscosity; }
  void setDensity( real _density ) { m_inputDensity = _density; }

  int vxIdx(int i, int j) const { return j*m_rowVelocity.x+i; }
  int vyIdx(int i, int j) const { return j*m_rowVelocity.y+i; }
  int cIdx(int i, int j) const { return j*m_gridSize.x+i; }

  int getcolVelX() const { return m_columnVelocity.x; }
  int getColVelY() const { return m_columnVelocity.y; }
  int getRowVelX() const { return m_rowVelocity.x; }
  int getRowVelY() const { return m_rowVelocity.y; }
  int getRowCell() const { return m_gridSize.x; }
  int getColCell() const { return m_gridSize.y; }
  int getTotCell() const { return m_totCell; }
  int getTotVelX() const { return m_totVelX; }
  int getTotVelY() const { return m_totVelY; }

protected:
  bool m_active = false;
  int m_totCell;
  int m_totVelX;
  int m_totVelY;
  real m_timeStep;
  real m_diffusion;
  real m_viscosity;
  real m_inputDensity;

  real * m_density;
  real * m_previousDensity;
  real * m_divergence;
  real * m_pressure;
  tuple<unsigned int> m_gridSize;
  tuple<unsigned int> m_rowVelocity;
  tuple<unsigned int> m_columnVelocity;
  tuple<real> m_min;
  tuple<real> m_max;
  tuple<real *> m_velocity;
  tuple<real *> m_previousVelocity;
  tuple<real> * m_pvx;
  tuple<real> * m_pvy;
};

#endif // SOLVER_H
