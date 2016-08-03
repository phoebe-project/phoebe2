#if !defined(__mesh_h)
#define __mesh_h

/*
  Library concerning routines for creation and manipulation of 
  triangular grid. 
  
  Author: Martin Horvat,  July 2016 
*/
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#include "utils/utils.h"

#include "triang/triang_marching.h"       // Maching triangulation
#include "triang/bodies.h"                // Definitions of different potentials
#include "eclipsing/eclipsing.h"          // Eclipsing/Hidden surface removal
#include "povray/povray.h"                // Exporting meshes to povray (minimalistic)
#include "reflection/reflection.h"        // Dealing with reflection effects/radiosity problem

#endif
