
#ifndef RAND_CPU_H
#define RAND_CPU_H

#include "Tuple.h"
#include <vector>

namespace Rand_CPU
{
    /// Consistent API for creating a std::vector of random floats
    int randFloats(std::vector<real>&);
}

#endif //RAND_CPU_H
