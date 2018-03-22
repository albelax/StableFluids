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

static void CPU_projection( benchmark::State& state ) //
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

static void GPU_projection( benchmark::State& state ) //
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

BENCHMARK_MAIN();

///----------------------------------------------------------------------------------------------
