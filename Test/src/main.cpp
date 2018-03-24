#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

//tests
#include "constructor.h"
#include "velocity.h"
#include "density.h"
#include "pressure.h"
#include "divergence.h"
// end tests

int main( int argc, char **argv )
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
