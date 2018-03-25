#include <QCoreApplication>
#include <benchmark/benchmark.h>
#include "rand_cpu.h"
#include "rand_gpu.h"
#include "GpuSolver.h"
#include "MacStableSolver.h"

///----------------------------------------------------------------------------------------------

static void Creation_of_a_string(benchmark::State& state)
{
  for (auto _ : state)
    std::string empty_string;
}

BENCHMARK(Creation_of_a_string);

///----------------------------------------------------------------------------------------------

static void CPU_solverCreation( benchmark::State& state )
{
  for ( auto _ : state )
    benchmark::DoNotOptimize( StableSolverCpu() );
}
BENCHMARK(CPU_solverCreation);

///----------------------------------------------------------------------------------------------

static void GPU_solverCreation( benchmark::State& state )
{
  for ( auto _ : state )
    benchmark::DoNotOptimize( GpuSolver() );
}
BENCHMARK(GPU_solverCreation);

///----------------------------------------------------------------------------------------------

static void CPU_solverActivation( benchmark::State& state )
{
  for ( auto _ : state )
  {
    StableSolverCpu solver;
    solver.activate();
  }
}
BENCHMARK(CPU_solverActivation);

///----------------------------------------------------------------------------------------------

static void GPU_solverActivation( benchmark::State& state )
{
  for ( auto _ : state )
  {
    GpuSolver solver;
    solver.activate();
  }
}
BENCHMARK(GPU_solverActivation);

///----------------------------------------------------------------------------------------------

static void CPU_projection( benchmark::State& state )
{
  StableSolverCpu solver;
  solver.activate();
  solver.randomizeArrays();
  for ( auto _ : state )
  {
    solver.projection();
  }

}
BENCHMARK(CPU_projection);

///----------------------------------------------------------------------------------------------

static void GPU_projection( benchmark::State& state )
{
  GpuSolver solver;
  solver.activate();
  solver.randomizeArrays();
  for ( auto _ : state )
  {
    solver.projection();
  }

}
BENCHMARK(GPU_projection);

///----------------------------------------------------------------------------------------------

static void CPU_advectVelocity( benchmark::State& state )
{
  StableSolverCpu solver;
  solver.activate();
  solver.randomizeArrays();
  for ( auto _ : state )
  {
    solver.advectVel();

  }
}
BENCHMARK(CPU_advectVelocity);

///----------------------------------------------------------------------------------------------


static void GPU_advectVelocity( benchmark::State& state )
{
  GpuSolver solver;
  solver.activate();
  solver.randomizeArrays();
  for ( auto _ : state )
  {
    solver.advectVelocity();

  }
}
BENCHMARK(GPU_advectVelocity);

///----------------------------------------------------------------------------------------------


static void CPU_advectCell( benchmark::State& state )
{
  StableSolverCpu solver;
  solver.activate();
  solver.randomizeArrays();
  for ( auto _ : state )
  {
    solver.advectCell( solver.m_density, solver.m_previousDensity );

  }
}
BENCHMARK( CPU_advectCell );

///----------------------------------------------------------------------------------------------


static void GPU_advectCell( benchmark::State& state )
{
  GpuSolver solver;
  solver.activate();
  solver.randomizeArrays();
  for ( auto _ : state )
  {
    solver.advectCell();

  }
}
BENCHMARK( GPU_advectCell );

///----------------------------------------------------------------------------------------------

static void CPU_diffuseVelocity( benchmark::State& state )
{
  StableSolverCpu solver;
  solver.activate();
  solver.randomizeArrays();
  for ( auto _ : state )
  {
    solver.diffuseVel();
  }

}
BENCHMARK(CPU_diffuseVelocity);

///----------------------------------------------------------------------------------------------
BENCHMARK_MAIN();

///----------------------------------------------------------------------------------------------
