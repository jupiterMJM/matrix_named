# Named Matrix
## Introduction
This file contains a class that allows you to link directly a matrix numpy to axis. It can be really useful when you deal with a matrix that represents a double entries table.

This class has been developped through my previous works where I needed to plot easily big matrix with given axis (ticks and labels), perform easily some operations on the axis (tranformation, conversion, reverse) and keeping the link matrix-axis.

This module is still under construction and I will add functionnalities whenever I need them. But, if you find any errors and/or think about new functionnalities, feel free to let a comment on github and I ll see it.

## What you can do
Below, you will find a list (that should be up-to-date) of what you can do with this class.
- have a matrix in 2 dimensions of shape n*m with an abscissa of m terms and an ordinate axis of n terms
- apply a function on one axis and keep the relation between the axis and the matrix values (conversion eg)
- apply a function on the matrix
- plot the matrix with the right ticks and labels automatically (3d plot are also possible)
- crop the matrix under one axis to keep only the part of the matrix you are interested in
- sum along axis
- select values/rows/colums based on a coherent given value on one axis