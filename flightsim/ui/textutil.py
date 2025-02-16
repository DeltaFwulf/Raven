"""Utility functions for terminal outputs"""

import numpy as np
from math import floor, log10

def formatArray(arr:np.array, sigFigs:int, tabs:int=0):
    """
    Prints an array in a human-readable format.

    sigFigs specifies the number of trailing digits to print, and tabs specifies the number of indents the array is printed from the left of the terminal
    """

    # TODO: allow for position as a number, rather than only as tabs

    sf = "%." + str(sigFigs) + "f"

    def getMaxOrder(array:np.array) -> int:

        maxMag = 0
        nonZeros = array[array != 0]
        
        for i in range(0, nonZeros.size):
            thisMag = floor(log10(nonZeros[i]))
            if thisMag > maxMag:
                maxMag = thisMag

        return maxMag
    
    # each row looks like this: [el1, el2, el3, ..., eln]
    rows, cols = np.shape(arr)
    pad = "{0: >" + str(getMaxOrder(arr) + 2 + sigFigs) + "}"

    arrayString = ""

    for i in range(0, rows):
        # print the next row:
        rowString = ""

        for i in range(0, tabs):
            rowString += "\t"
        
        rowString += "["
        
        for j in range(0, cols):
            rowString += pad.format(sf % arr[i,j]) + (", " if j < cols - 1 else "]")

        arrayString += rowString + ("\n" if i < rows - 1 else "")

    return arrayString
                          
        