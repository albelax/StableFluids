#ifndef _PARAMETERS_H
#define _PARAMETERS_H

namespace Common 
{
  int multiplier = 1;
  int gridWidth = 32 * multiplier;
  int gridHeight = 32 * multiplier;

  int totCells = gridWidth * gridHeight;

  int rowVelocityX = gridWidth + 1;
  int rowVelocityY = gridWidth;

  int columnVelocityX = gridHeight;
  int columnVelocityY = gridHeight + 1;

  int totHorizontalVelocity = rowVelocityX * columnVelocityX;
  int totVerticalVelocity = rowVelocityY * columnVelocityY;
}

#endif // _PARAMETERS_H
