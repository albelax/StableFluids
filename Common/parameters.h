#ifndef _PARAMETERS_H
#define _PARAMETERS_H

namespace Common 
{
  unsigned int multiplier = 1;
  unsigned int gridWidth = 128 ;
  unsigned int gridHeight = 128;

  unsigned int totCells = gridWidth * gridHeight;

  unsigned int rowVelocityX = gridWidth + 1;
  unsigned int rowVelocityY = gridWidth;

  unsigned int columnVelocityX = gridHeight;
  unsigned int columnVelocityY = gridHeight + 1;

  unsigned int totHorizontalVelocity = rowVelocityX * columnVelocityX;
  unsigned int totVerticalVelocity = rowVelocityY * columnVelocityY;
}

#endif // _PARAMETERS_H
