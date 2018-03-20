#include <QCoreApplication>
#include <benchmark/benchmark.h>
#include "GpuSolver.h"
#include "MacStableSolver.h"

///----------------------------------------------------------------------------------------------

static void BM_StringCreation(benchmark::State& state)
{
  for (auto _ : state)
    std::string empty_string;
}

BENCHMARK(BM_StringCreation);

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

static void CPU_solverActivatiopn( benchmark::State& state )
{
  for ( auto _ : state )
  {
    StableSolverCpu solver;
    solver.activate();
  }

}
BENCHMARK(CPU_solverActivatiopn);

///----------------------------------------------------------------------------------------------

static void GPU_solverActivatiopn( benchmark::State& state )
{
  for ( auto _ : state )
  {
    GpuSolver solver;
    solver.activate();
  }

}
BENCHMARK(GPU_solverActivatiopn);

///----------------------------------------------------------------------------------------------

BENCHMARK_MAIN();

///----------------------------------------------------------------------------------------------
