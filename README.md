# A lightweight package for generating continuous random fields
This repo contains several examples of gaussian fields created using the ```convolved-noise``` Python package, as well as the package's source file.

## Installation
The ```convolved-noise``` package is available from PyPi, via 
```
pip install convolved-noise
```

## Basic usage
Once ```convolved-noise``` is installed, run ```from cnvlnoise import noise``` to access the core method providing the package's essential functionality. 
Calling ```noise``` results in a numpy array containing the values of a gaussian process on a regular grid. The only argument needed is the shape of the 
grid--for instance, ```noise((5, 5))``` will produce a 5x5 array of correlated random values.
