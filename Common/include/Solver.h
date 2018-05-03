#ifndef SOLVER_H
#define SOLVER_H

#include "tuple.h"

#define TESTING 1

#define SWAP(value0,value) { real *tmp = value0; value0 = value; value = tmp; }

/// \brief The Solver class, pure virtual class, it can't be instantiated,
/// both cpu and gpu solvers will inherit from this one


class Solver
{
public:
  Solver() {}
  ~Solver() {}

  /// \brief cleanBuffer, resets previous density and previous velocities
  virtual void cleanBuffer() = 0;

  /// \brief activate allocates memory
  virtual void activate() = 0;

  /// \brief setVel0, sets the velocity from the mouse input
  /// \param i, destination index
  /// \param j, destination index
  /// \param _vx0, previous mouse position in the x
  /// \param _vy0 previous mouse position in the y
  virtual void setVel0(int i, int j, real _vx0, real _vy0) = 0;

  /// \brief setD0, sset density from mouse input
  /// \param i, destination index
  /// \param j, destination index
  virtual void setD0(int i, int j ) = 0;

  /// \brief addSource, adds density and velocity after the user input
  virtual void addSource() = 0;

  /// \brief animVel, velocity step
  virtual void animVel() = 0;

  /// \brief animDen, density step
  virtual void animDen() = 0;

  /// \brief reset, resets density and velocities
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

#if !TESTING
protected:
#endif
  /// \brief m_active, true if the solver is currently in use, it is set true after
  /// the memory allocations, in case is false the destructor will not free the memory
  bool m_active = false;

  /// \brief m_totCell, number of cells, or pixels, in the grid
  int m_totCell;
  /// \brief m_totVelX, number of cells containing the horizontal velocity
  int m_totVelX;
  /// \brief m_totVelY, number of cells containing the vertical velocity
  int m_totVelY;
  /// \brief m_timeStep, timestep, or playback speed, of the simulation, changed from input
  real m_timeStep;
  /// \brief m_diffusion, amount of diffusion, changed from input
  real m_diffusion;
  /// \brief m_viscosity, amount of viscosity, changed from input
  real m_viscosity;
  /// \brief m_inputDensity, amount of density, changed from mouse input
  real m_inputDensity;
  /// \brief m_density, array storing the density in each cell
  real * m_density;
  /// \brief m_previousDensity, array storing the previous density in each cell
  real * m_previousDensity;
  /// \brief m_divergence, array storing the divergence in each cell
  real * m_divergence;
  /// \brief m_pressure, array storing the pressure in each cell
  real * m_pressure;
  /// \brief m_gridSize, width and height of the grid
  tuple<unsigned int> m_gridSize;
  /// \brief m_rowVelocity, width and height of the horizontal velocity
  tuple<unsigned int> m_rowVelocity;
  /// \brief m_columnVelocity, width and height of the vertical velocity
  tuple<unsigned int> m_columnVelocity;
  tuple<real> m_min;
  tuple<real> m_max;
  /// \brief m_velocity, velocity in x and y for each cell
  tuple<real *> m_velocity;
  /// \brief m_previousVelocity, previous velocity in x and y for each cell
  tuple<real *> m_previousVelocity;
  tuple<real> * m_pvx;
  tuple<real> * m_pvy;
};

#endif // SOLVER_H
