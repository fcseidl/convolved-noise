# A lightweight package for generating continuous random fields
This repo contains several examples of gaussian fields created using the ```convolved-noise``` Python package, as well as the package's source file.

## Installation
The ```convolved-noise``` package is available from PyPi, via 
```
pip install convolved-noise
```
It is also defined fully by the ```noise.py``` file in this repository.

## Basic usage
The main functionality of ```convolved-noise``` is provided by a single method, called ```noise```, which returns a numpy array containing the values 
of a gaussian process on a regular grid.
