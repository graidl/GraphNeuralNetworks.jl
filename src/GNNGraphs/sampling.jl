"""
    sample_neighbors(g, nodes, K=-1; dir=:in, replace=false, dropnodes=false)

Sample neighboring edges of the given nodes and return the induced subgraph.
For each node, a number of inbound (or outbound when `dir = :out``) edges will be randomly chosen. 
If `dropnodes=false`, the graph returned will then contain all the nodes in the original graph, 
but only the sampled edges.

The returned graph will contain an edge feature `EID` corresponding to the id of the edge
in the original graph. If `dropnodes=true`, it will also contain a node feature `NID` with
the node ids in the original graph.

# Arguments

- `g`. The graph.
- `nodes`. A list of node IDs to sample neighbors from.
- `K`. The maximum number of edges to be sampled for each node.
       If -1, all the neighboring edges will be selected.
- `dir`. Determines whether to sample inbound (`:in`) or outbound (``:out`) edges (Default `:in`).
- `replace`. If `true`, sample with replacement.
- `dropnodes`. If `true`, the resulting subgraph will contain only the nodes involved in the sampled edges.
     
# Examples

```julia
julia> g = rand_graph(20, 100)
GNNGraph:
    num_nodes = 20
    num_edges = 100

julia> sample_neighbors(g, 2:3)
GNNGraph:
    num_nodes = 20
    num_edges = 9
    edata:
        EID => (9,)

julia> sg = sample_neighbors(g, 2:3, dropnodes=true)
GNNGraph:
    num_nodes = 10
    num_edges = 9
    ndata:
        NID => (10,)
    edata:
        EID => (9,)

julia> sg.ndata.NID
10-element Vector{Int64}:
  2
  3
 17
 14
 18
 15
 16
 20
  7
 10

julia> sample_neighbors(g, 2:3, 5, replace=true)
GNNGraph:
    num_nodes = 20
    num_edges = 10
    edata:
        EID => (10,)
```
"""
function sample_neighbors(g::GNNGraph{<:COO_T}, nodes, K=-1; 
        dir=:in, replace=false, dropnodes=false)
    @assert dir ∈ (:in, :out)
    _, eidlist = adjacency_list(g, nodes; dir, with_eid=true)
    for i in 1:length(eidlist)
        if replace 
            k = K > 0 ? K : length(eidlist[i])
        else
            k = K > 0 ? min(length(eidlist[i]), K) : length(eidlist[i])
        end
        eidlist[i] = StatsBase.sample(eidlist[i], k; replace)
    end
    eids = reduce(vcat, eidlist)
    s, t = edge_index(g)
    w = get_edge_weight(g)
    s = s[eids]
    t = t[eids]
    w = isnothing(w) ? nothing : w[eids]

    edata = getobs(g.edata, eids)
    edata.EID = eids

    num_edges = length(eids)

    if !dropnodes
        graph = (s, t, w)
    
        gnew = GNNGraph(graph, 
                    g.num_nodes, num_edges, g.num_graphs,
                    g.graph_indicator,
                    g.ndata, edata, g.gdata)
    else    
        nodes_other = dir == :in ? setdiff(s, nodes) : setdiff(t, nodes)
        nodes_all = [nodes; nodes_other]
        nodemap = Dict(n => i for (i, n) in enumerate(nodes_all))
        s = [nodemap[s] for s in s]
        t = [nodemap[t] for t in t]
        graph = (s, t, w)
        graph_indicator = g.graph_indicator !== nothing ? g.graph_indicator[nodes_all] : nothing
        num_nodes = length(nodes_all)
        ndata = getobs(g.ndata, nodes_all)
        ndata.NID = nodes_all        

        gnew = GNNGraph(graph, 
                    num_nodes, num_edges, g.num_graphs,
                    graph_indicator,
                    ndata, edata, g.gdata)
    end
    return gnew
end
