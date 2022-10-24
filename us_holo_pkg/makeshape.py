import numpy as np
import math


class notdefinederror(Exception):
    print('Disc center must be within grid')

def makeDisc(Nx, Ny, cx, cy, radius):
    # Creates a binary map of a filled disck within a 2D grid (pre-defined: Nx, Ny)
    # Input:
    #       Nx,Ny: size of the 2D grid
    #       cx,cy: center of the disc
    #       radius: raidus of disc
    # Output:
    #       disc: 2D binary map of a filled disc



    Nx=round(Nx)
    Ny=round(Ny)
    cx=round(cx)
    cy=round(cy)

    if cx==0:
        cx=math.floor(Nx/2)+1
    if cy==0:
        cy=math.floor(Ny/2)+1

    # check the inputs
    if cx < 1 or cx > Nx or cy<1 or cy>Ny:
        raise notdefinederror

    # define pixel map
    X,Y=np.ogrid[:Nx,:Ny]
    dist_from_center=np.sqrt((X-cx)**2+(Y-cy)**2)


    #create disc
    disc=dist_from_center<=radius



    return disc