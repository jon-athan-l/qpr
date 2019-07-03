# qpr
Particle track reconstruction at LBNL.

Make sure you create a virtual environment and run setup.py in qpr/qallse (QPR's engine) to set up all the necessary files!
And then activate the virtual environment to work in your own mirror world :)


*A high-level overview*

AMBIGUITY: runs ambiguity resolution on a set of reconstructed tracks.

QALLSE: reconstructs a set of candidate tracks based on raw particle collider input, using quantum annealing to solve the NP-hard QUBO (Quadratic Unconstrained Binary Optimization) problem.

DATA: relevant bits of data, including plots and simulated collider data (so we have an "objective truth" to compare against).
