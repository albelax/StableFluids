#ifndef _PARAMETERS_H
#define _PARAMETERS_H

namespace Common 
{
  int multiplier = 4;
  int gridWidth =  128 * multiplier;
  int gridHeight = 128 * multiplier;

  int totCells = gridWidth * gridHeight;

  int rowVelocityX = gridWidth + 1;
  int rowVelocityY = gridWidth;

  int columnVelocityX = gridHeight;
  int columnVelocityY = gridHeight + 1;

  int totHorizontalVelocity = rowVelocityX * columnVelocityX;
  int totVerticalVelocity = rowVelocityY * columnVelocityY;
}

#endif // _PARAMETERS_H
