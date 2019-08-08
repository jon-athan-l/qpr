# qpr
Particle track reconstruction at LBNL.

## Setup

Create a virtual environment and run setup.py in `qpr/qallse` (QPR's engine) to set up all the necessary files!
And then activate the virtual environment to work in your own mirror world :)

Running Qallse, which generates potential reconstructed tracks and the input to the ambiguity resolution, can best be explained by [its creator](https://github.com/derlin/hepqpr-qallse#setup-and-usage) ;).

Generate some input tracks from Qallse, and then start working in the `qpr/ambiguity` directory.

To run the ambiguity resolution, run the command `python run.py` while cd'd into the `qpr/ambiguity` directory.

To generate plots about your input track properties, open up `qpr/ambiguity/propertyplot.py` and in the build section, create
an Engine instance and call whatever command you like:
```
>>> e = Engine()
>>> e.find_reconstructed_nhits()
```
A list of properties you can generate distributions of:

`find_spatial()`, which tells you all about the spatial coordinates of the track. These coordinates include `theta`, `eta`, `phi`, and `pT`.

`find_shared_hits()`, which finds the number of shared tracks and shared hits for each track. "Phi slicing" is optionally implemented to narrow a field of the track's neighbors by including `phi_slice=[angle]`, where `[angle]` is used to find the radius of the phi slice, `pi/[angle]`. Another parameter, `remember=True`, can be optionally specified as a form of memoization. The memoization creates two new attributes for each track, `track.shared_tracks` and `track.shared_hits`, which are Python lists that contain other track IDs. This can be potentially dangerous because it is mutative, and adds on this algorithm's finished output to each track in the track dataset. This is included to help speed up the runtime of the algorithm.

`find_purity_recall()`, which finds the purity and efficiency of each track. Efficiency is the number of "real" reconstructed hits that are matched out of all the hits that were reconstructed to a track, and purity is the number of "real" reconstructed hits that are matched over how many *should* be matched to the track.

`find_reconstructed_nhits()`, which finds the "completeness" of a track, defined by each track's number of hits and holes (missing hits).

Verbosity can also be specified in any of the above functions by including the parameter `verbose=True` to see printed outputs of the algorithms while they run, for fun :).

You can also specify custom paths to different datasets by including more parameters in the `Engine` constructor: `Engine([custom source path to reconstructed data, true hit data, true particle data])`

## A high-level overview

**Ambiguity**: runs ambiguity resolution on a set of reconstructed tracks. Uses quantum annealing to solve the NP-hard QUBO (Quadratic Unconstrained Binary Optimization) problem of classifying "true" tracks.

**Qallse**: reconstructs a set of candidate tracks based on raw particle collider input, using quantum annealing to solve the NP-hard QUBO (Quadratic Unconstrained Binary Optimization) problem of classifying "true" doublets.

**Data**: relevant bits of data, including plots and simulated collider data (so we have an "objective truth" to compare against).

## Quantum Annealing

Solves the NP-hard QUBO (Quadratic Unconstrained Binary Optimization) problem (finding the global maximum of a quadratic function). Shares a striking similarity and identical functionality with a GNN (Graph Neural Network), a technique used in deep learning in which a directed graph is built and weights are assigned to its nodes, which modify the data that pass through them. The node weights are updated many times. Eventually the weights become accurate and the graph is able to successfully predict outcomes that match the context of the dataset it is trained on. Graph neural networks lie in the intersection of graph theory and machine learning.

This process is called "deep learning" because in a GNN the weights are often modified to such an extent that the creator of the graph does not know (and may even not care) what is inside his graph. They only know what goes in and comes out.

In a quantum annealer, weights are assigned to electrons and their energy states. A physical principle of electrons is that they try to sink to the lowest energy state, and can even exhibit quantum tunneling to do so. The electrons that "drop" to the lowest energy state are the quantum oracle's answer (think about pinballs in a pinball machine falling down). Every electron tries to drop at the same time, "globally". Computer scientists like this because it means that the processing time is independent of how large a dataset you are working with.

An adjacency matrix is created to capture the graph in matrix form (this moves it into the domain of linear algebra, which we are good at). The quantum GNN uses *two* of these "flattened" graphs. For each track it examines, the track is associated with two categories of weights--a weight of the track's own quality (the bias term) and weights that compare every track to another track (the interaction, or coupling term).

Fascinatingly, the quantum annealer is then able to model the *interactions* between the tracks.

Very cool!


### Contributors
**Lucy Linder**, who wrote the incredible Qallse engine;
**Alex Smith**, who worked on the ambiguity resolution before me and wrote the backbone of the quantum annealing code;
**Paolo Calafiura**, my advisor at LBNL and a wonderful person to get advice from; and 
**Heather Gray**, my advisor at LBNL and who helped with much of the theory and design behind the project.
