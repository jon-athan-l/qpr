# qpr
Particle track reconstruction at LBNL.

Make sure you create a virtual environment and run setup.py in qpr/qallse (QPR's engine) to set up all the necessary files!
And then activate the virtual environment to work in your own mirror world :)


## A high-level overview

**Ambiguity**: runs ambiguity resolution on a set of reconstructed tracks. Uses quantum annealing to solve the NP-hard QUBO (Quadratic Unconstrained Binary Optimization) problem of classifying "true" tracks.

**Qallse"": reconstructs a set of candidate tracks based on raw particle collider input, using quantum annealing to solve the NP-hard QUBO (Quadratic Unconstrained Binary Optimization) problem of classifying "true" doublets.

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
