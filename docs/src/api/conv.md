```@meta
CurrentModule = GraphNeuralNetworks
```

# Convolutional Layers

Many different types of graphs convolutional layers have been proposed in the literature. Choosing the right layer for your application could involve a lot of exploration. 
Some of the most commonly used layers are the [`GCNConv`](@ref) and the [`GATv2Conv`](@ref). Multiple graph convolutional layers are typically stacked together to create a graph neural network model
(see [`GNNChain`](@ref)).

The table below lists all graph convolutional layers implemented in the *GraphNeuralNetworks.jl*. It also highlights the presence of some additional capabilities with respect to basic message passing:
- *Sparse Ops*: implements message passing as multiplication by sparse adjacency matrix instead of the gather/scatter mechanism. This can lead to better cpu performances but it is not supported on gpu yet. 
- *Edge Weights*: supports scalar weights (or equivalently scalar features) on edges. 
- *Edge Features*: supports feature vectors on edges.

| Layer                       |Sparse Ops|Edge Weight|Edge Features| 
| :--------                   |  :---:   |:---:      |:---:        |               
| [`AGNNConv`](@ref)          |          |           |    ✓        |
| [`CGConv`](@ref)            |          |           |             |
| [`ChebConv`](@ref)          |          |           |             |
| [`EdgeConv`](@ref)          |          |           |             |
| [`EGNNConv`](@ref)          |          |           |     ✓       |
| [`GATConv`](@ref)           |          |           |     ✓       |
| [`GATv2Conv`](@ref)         |          |           |     ✓       |
| [`GatedGraphConv`](@ref)    |     ✓    |           |             |
| [`GCNConv`](@ref)           |     ✓    |     ✓     |             | 
| [`GINConv`](@ref)           |     ✓    |           |             |
| [`GMMConv`](@ref)           |          |           |     ✓       |
| [`GraphConv`](@ref)         |     ✓    |           |             |
| [`MEGNetConv`](@ref)        |          |           |     ✓       |
| [`NNConv`](@ref)            |          |           |     ✓       |
| [`ResGatedGraphConv`](@ref) |          |           |             |
| [`SAGEConv`](@ref)          |     ✓    |           |             |
| [`SGConv`](@ref)            |     ✓    |           |             |


## Docs

```@autodocs
Modules = [GraphNeuralNetworks]
Pages   = ["layers/conv.jl"]
Private = false
```
