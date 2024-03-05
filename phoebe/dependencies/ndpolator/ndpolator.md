ndpolator: fast, n-dimensional linear interpolation and extrapolation on sparse grids {#mainpage}
=====================================================================================

Introduction
------------

Ndpolator allows you to interpolate and/or extrapolate function values on an n-dimensional cartesian grid with missing values. Given a sequence of axes, ndpolator successively reduces the number of dimensions in which it interpolates and/or extrapolates. It starts with the corners of the ``N``-dimensional hypercube to obtain interpolated values in the ``(N-1)``-dimensional hyperplane while keeping the last axis constant. It then removes the last axis and forms an ``(N-1)``-dimensional hypercube from the interpolants. The process is then restarted and continued until we obtain the interpolant in the point of interest.

Overall logic
-------------

Ndpolator requires the following structures as input:

* a set of axes that span the n-dimensional grid;
* an n-dimensional matrix of function values on the grid spanned by the axes;
* a set of query points for which we want to determine their function values.

Grid points are given by a cartesian product of axial vertices. For example, axes ``a1 = [1, 2, 3, 4]`` and ``a2 = [5, 6, 7]`` span a 2-dimensional grid; values ``1, 2, 3, 4`` are *vertices* on ``a1``; values ``5, 6, 7`` are *vertices* on ``a2``. Their cartesian product defines 12 grid points: ``(1, 5)``, ``(1, 6)``, ``(1, 7)``, ``(2, 5)``, ``(2, 6)``, ``(2, 7)``, ``(3, 5)``, ``(3, 6)``, ``(3, 7)``, ``(4, 5)``, ``(4, 6)`` and ``(4, 7)``. If a grid point has a defined function value, it is called a *node*. If a grid point does not have a defined function value, it is called a *void*. A *sparse grid* is a grid that has a combination of nodes and voids. Nodes can hold scalars or arrays of any length; voids hold ``nan``s. The job of the ndpolator is to linearly interpolate and/or extrapolate function values on the sparse grid.

This is done in several steps. For each query point, the interpolator:

1. finds the local n-dimensional hypercube that encloses the query point, and stores the superior corner's indices;
2. transforms the query point into hypercube-normalized coordinates, ``[0, 1] x [0, 1] x ... x [0, 1] = [0, 1]^n``;
3. flags each query point component as on-vertex, on-grid, or out-of-bounds;
4. for any on-vertex components, it reduces hypercube dimensionality;
5. checks to see if the query point is inside of a fully defined hypercube; if so, it proceeds with interpolation;
6. if the query point is not in a fully defined hypercube, it extrapolates from the nearest node or the nearest fully defined hypercube.

Continuing with our example above, say that ``q1 = [2.5, 6.2]`` is the query point; the enclosing 2-dimensional hypercube is ``hc1 = {(1, 1), (1, 2), (2, 1), (2, 2)}`` (these are the boundary indices along ``a1`` and ``a2``). The reference to that hypercube are the indices of the superior corner, ``sc1 = (2, 2)``. The query point is then transformed into the hypercube-normalized coordinates, ``nq1 = [(2.5-2)/(3-2), (6.2-6)/(7-6)] = [0.5, 0.2]``. This achieves equal step sizes along every direction, irrespective of the absolute vertex values. Both components are then flagged as on-grid, and as no components coincide with any vertices, function values are interpolated and returned.

Let us consider another example: let ``q2 = [2.0, 6.5]``. This time the first component coincides with a vertex, and the dimension of the hypercube can be reduced to 1-D: ``hc2 = {(1, 1), (1, 2)}``. The original hypercube-normalized coordinate, ``nq2 = [0, 0.5]``, is then reduced to ``nq2 = [0.5]``. Interpolation then continues as before, this time in 1 dimension instead of 2.

Finally, let us consider a query point off the grid: ``q3 = [0.8, 5.3]``. The corresponding normalized query point is ``nq3 = [-0.2, 0.3]`` and the enclosing hypercube does not exist. There are 3 extrapolation types: no extrapolation, extrapolation to the nearest node, and linear extrapolation. If no extrapolation is requested, a nan is returned. If extrapolation to the nearest node is requested, ndpolator finds the nearest node, in this case ``(0, 0)``, and it assigns its function value to the ndpolant. If linear extrapolation is requested, ndpolator finds the nearest fully defined hypercube, in this case ``(1, 1)``, and linearly extrapolates from that hypercube.

Interpolation implementation details
------------------------------------

The algorithm takes an array of query points ``x``, an ``N``-dimensional array of inferior vertices ``lo``, an ``N``-dimensional array of superior vertices ``hi``, and an array of 2<sup>N</sup> function values ``fv``, sorted in the native C order:

@f[
    fv = \left[
        \begin{array}{c}
            f(x_{0,\mathrm{lo}}, \dots, x_{N-1, \mathrm{lo}}, x_{N, \mathrm{lo}}) \\
            f(x_{0,\mathrm{lo}}, \dots, x_{N-1, \mathrm{lo}}, x_{N, \mathrm{hi}}) \\
            f(x_{0,\mathrm{lo}}, \dots, x_{N-1, \mathrm{hi}}, x_{N, \mathrm{lo}}) \\
            f(x_{0,\mathrm{lo}}, \dots, x_{N-1, \mathrm{hi}}, x_{N, \mathrm{hi}}) \\
            \vdots \\
            f(x_{0,\mathrm{hi}}, \dots, x_{N-1, \mathrm{hi}}, x_{N, \mathrm{lo}}) \\
            f(x_{0,\mathrm{hi}}, \dots, x_{N-1, \mathrm{hi}}, x_{N, \mathrm{hi}}) \\
        \end{array}
    \right],
@f]

where @f$x_k@f$ are the function values for the ``k``-th axis. Interpolation proceeds from the last axis to the first, and array ``fv`` is modified in the process.

Note that the choice of interpolating along the last axis is both arbitrary (we could just as easily choose any other axis to start with) and general (we could pivot the axes if another interpolation sequence is desired). For as long as the parameter space is sufficiently linear, the choice of axis sequence is not too important, but any local non-linearity will cause the sequence to matter.

Summary of ndpolator's terminology
----------------------------------

| **Term**                  | **Definition**                                                                            |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| **vertex**                | a real value that denotes a position on the axis                                          |
| **axis**                  | an array of vertices                                                                      |
| **basic axis**            | an axis that spans the sparse grid (i.e., grid points can be nodes or voids)              |
| **attached axis**         | an axis that spans the full grid (i.e., all grid points are guaranteed to be nodes)       |
| **axes**                  | a set of ``n`` axes that span the ``n``-dimensional grid; axes can have different lengths |
| **grid point**            | a combination of vertices from each axis; one of the cartesian product elements           |
| **function value**        | a number or an array that corresponds to a grid point                                     |
| **node**                  | grid point with a defined function value                                                  |
| **void**                  | grid point without a defined function value                                               |
| **query point**           | point of interest, on- or off-grid, where a function value is sought                      |
| **query point component** | a single component of the n-dimensional query point                                       |
| **hypercube**             | an n-dimensional subgrid that encloses the query point                                    |
| **superior corner**       | hypercube grid point with superior (highest) axis indices                                 |
| **fdhc**                  | fully defined hypercube: a hypercube with all grid points being nodes                     |
| **ndpolant**              | linearly interpolated or extrapolated function value at a query point                     |
