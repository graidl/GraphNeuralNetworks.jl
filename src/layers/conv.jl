@doc raw"""
    GCNConv(in => out, σ=identity; [bias, init, add_self_loops, use_edge_weight])

Graph convolutional layer from paper [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

Performs the operation
```math
\mathbf{x}'_i = \sum_{j\in N(i)} a_{ij} W \mathbf{x}_j
```
where ``a_{ij} = 1 / \sqrt{|N(i)||N(j)|}`` is a normalization factor computed from the node degrees. 

If the input graph has weighted edges and `use_edge_weight=true`, than ``a_{ij}`` will be computed as
```math
a_{ij} = \frac{e_{j\to i}}{\sqrt{\sum_{j \in N(i)}  e_{j\to i}} \sqrt{\sum_{i \in N(j)}  e_{i\to j}}}
```

The input to the layer is a node feature array `X` of size `(num_features, num_nodes)`
and optionally an edge weight vector.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `σ`: Activation function. Default `identity`.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. Default `false`.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(3, g.num_nodes)

# create layer
l = GCNConv(3 => 5) 

# forward pass
y = l(g, x)       # size:  5 × num_nodes

# convolution with edge weights
w = [1.1, 0.1, 2.3, 0.5]
y = l(g, x, w)

# Edge weights can also be embedded in the graph.
g = GNNGraph(s, t, w)
l = GCNConv(3 => 5, use_edge_weight=true) 
y = l(g, x) # same as l(g, x, w) 
```
"""
struct GCNConv{W<:AbstractMatrix, B, F} <: GNNLayer
    weight::W
    bias::B
    σ::F
    add_self_loops::Bool
    use_edge_weight::Bool
end

@functor GCNConv

function GCNConv(ch::Pair{Int,Int}, σ=identity;
                 init=glorot_uniform, 
                 bias::Bool=true,
                 add_self_loops=true,
                 use_edge_weight=false)
    in, out = ch
    W = init(out, in)
    b = bias ? Flux.create_bias(W, true, out) : false
    GCNConv(W, b, σ, add_self_loops, use_edge_weight)
end

function (l::GCNConv)(g::GNNGraph, x::AbstractMatrix{T}, edge_weight::EW=nothing) where 
    {T, EW<:Union{Nothing,AbstractVector}}
    
    @assert !(g isa GNNGraph{<:ADJMAT_T} && edge_weight !== nothing) "Providing external edge_weight is not yet supported for adjacency matrix graphs"

    if edge_weight !== nothing
        @assert length(edge_weight) == g.num_edges "Wrong number of edge weights (expected $(g.num_edges) but given $(length(edge_weight)))" 
    end

    if l.add_self_loops
        g = add_self_loops(g)
        if edge_weight !== nothing
            # Pad weights with ones
            # TODO for ADJMAT_T the new edges are not generally at the end
            edge_weight = [edge_weight; fill!(similar(edge_weight, g.num_nodes), 1)]
            @assert length(edge_weight) == g.num_edges
        end
    end
    Dout, Din = size(l.weight)
    if Dout < Din
        # multiply before convolution if it is more convenient, otherwise multiply after
        x = l.weight * x
    end
    d = degree(g, T; dir=:in, edge_weight)
    c = 1 ./ sqrt.(d)
    x = x .* c'
    if edge_weight !== nothing
        x = propagate(e_mul_xj, g, +, xj=x, e=edge_weight)
    elseif l.use_edge_weight        
        x = propagate(w_mul_xj, g, +, xj=x)
    else
        x = propagate(copy_xj, g, +, xj=x)
    end
    x = x .* c'
    if Dout >= Din
        x = l.weight * x
    end
    return l.σ.(x .+ l.bias)
end

function (l::GCNConv)(g::GNNGraph{<:ADJMAT_T}, x::AbstractMatrix, edge_weight::AbstractVector)
    g = GNNGraph(edge_index(g)...; g.num_nodes)  # convert to COO
    return l(g, x, edge_weight)
end

function Base.show(io::IO, l::GCNConv)
    out, in = size(l.weight)
    print(io, "GCNConv($in => $out")
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end


@doc raw"""
    ChebConv(in => out, k; bias=true, init=glorot_uniform)

Chebyshev spectral graph convolutional layer from
paper [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375).

Implements

```math
X' = \sum^{K-1}_{k=0}  W^{(k)} Z^{(k)}
```

where ``Z^{(k)}`` is the ``k``-th term of Chebyshev polynomials, and can be calculated by the following recursive form:

```math
\begin{aligned}
Z^{(0)} &= X \\
Z^{(1)} &= \hat{L} X \\
Z^{(k)} &= 2 \hat{L} Z^{(k-1)} - Z^{(k-2)}
\end{aligned}
```

with ``\hat{L}`` the [`scaled_laplacian`](@ref).

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `k`: The order of Chebyshev polynomial.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct ChebConv{W<:AbstractArray{<:Number,3}, B} <: GNNLayer
    weight::W
    bias::B
    k::Int
end

function ChebConv(ch::Pair{Int,Int}, k::Int;
                  init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, in, k)
    b = bias ? Flux.create_bias(W, true, out) : false
    ChebConv(W, b, k)
end

@functor ChebConv

function (c::ChebConv)(g::GNNGraph, X::AbstractMatrix{T}) where T
    check_num_nodes(g, X)
    @assert size(X, 1) == size(c.weight, 2) "Input feature size must match input channel size."
    
    L̃ = scaled_laplacian(g, eltype(X))

    Z_prev = X
    Z = X * L̃
    Y = view(c.weight,:,:,1) * Z_prev
    Y += view(c.weight,:,:,2) * Z
    for k = 3:c.k
        Z, Z_prev = 2*Z*L̃ - Z_prev, Z
        Y += view(c.weight,:,:,k) * Z
    end
    return Y .+ c.bias
end

function Base.show(io::IO, l::ChebConv)
    out, in, k = size(l.weight)
    print(io, "ChebConv(", in, " => ", out)
    print(io, ", k=", k)
    print(io, ")")
end


@doc raw"""
    GraphConv(in => out, σ=identity; aggr=+, bias=true, init=glorot_uniform)

Graph convolution layer from Reference: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244).

Performs:
```math
\mathbf{x}_i' = W_1 \mathbf{x}_i + \square_{j \in \mathcal{N}(i)} W_2 \mathbf{x}_j
```

where the aggregation type is selected by `aggr`.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct GraphConv{W<:AbstractMatrix,B,F,A} <: GNNLayer
    weight1::W
    weight2::W
    bias::B
    σ::F
    aggr::A
end

@functor GraphConv

function GraphConv(ch::Pair{Int,Int}, σ=identity; aggr=+,
                   init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W1 = init(out, in)
    W2 = init(out, in)
    b = bias ? Flux.create_bias(W1, true, out) : false
    GraphConv(W1, W2, b, σ, aggr)
end

function (l::GraphConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    m = propagate(copy_xj, g, l.aggr, xj=x)
    x = l.σ.(l.weight1 * x .+ l.weight2 * m .+ l.bias)
    return x
end

function Base.show(io::IO, l::GraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1)-1)
    print(io, "GraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


@doc raw"""
    GATConv(in => out, [σ; heads, concat, init, bias, negative_slope, add_self_loops])
    GATConv((in, ein) => out, ...)

Graph attentional layer from the paper [Graph Attention Networks](https://arxiv.org/abs/1710.10903).

Implements the operation
```math
\mathbf{x}_i' = \sum_{j \in N(i) \cup \{i\}} \alpha_{ij} W \mathbf{x}_j
```
where the attention coefficients ``\alpha_{ij}`` are given by
```math
\alpha_{ij} = \frac{1}{z_i} \exp(LeakyReLU(\mathbf{a}^T [W \mathbf{x}_i; W \mathbf{x}_j]))
```
with ``z_i`` a normalization factor. 

In case `ein > 0` is given, edge features of dimension `ein` will be expected in the forward pass 
and the attention coefficients will be calculated as  
```math
\alpha_{ij} = \frac{1}{z_i} \exp(LeakyReLU(\mathbf{a}^T [W_e \mathbf{e}_{j\to i}; W \mathbf{x}_i; W \mathbf{x}_j]))
```

# Arguments

- `in`: The dimension of input node features.
- `ein`: The dimension of input edge features. Default 0 (i.e. no edge features passed in the forward).
- `out`: The dimension of output node features.
- `σ`: Activation function. Default `identity`.
- `bias`: Learn the additive bias if true. Default `true`.
- `heads`: Number attention heads. Default `1`.
- `concat`: Concatenate layer output or not. If not, layer output is averaged over the heads. Default `true`.
- `negative_slope`: The parameter of LeakyReLU.Default `0.2`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `true`.
"""
struct GATConv{DX<:Dense,DE<:Union{Dense,Nothing},T,A<:AbstractMatrix,F,B} <: GNNLayer
    dense_x::DX
    dense_e::DE
    bias::B
    a::A
    σ::F
    negative_slope::T
    channel::Pair{NTuple{2,Int}, Int}
    heads::Int
    concat::Bool
    add_self_loops::Bool
end

@functor GATConv
Flux.trainable(l::GATConv) = (l.dense_x, l.dense_e, l.bias, l.a)

GATConv(ch::Pair{Int,Int}, args...; kws...) = GATConv((ch[1], 0) => ch[2], args...; kws...)

function GATConv(ch::Pair{NTuple{2,Int},Int}, σ=identity;
                 heads::Int=1, concat::Bool=true, negative_slope=0.2,
                 init=glorot_uniform, bias::Bool=true, add_self_loops=true)
    (in, ein), out = ch    
    if add_self_loops
        @assert ein == 0 "Using edge features and setting add_self_loops=true at the same time is not yet supported."
    end
             
    dense_x = Dense(in, out*heads, bias=false)
    dense_e = ein > 0 ? Dense(ein, out*heads, bias=false) : nothing
    b = bias ? Flux.create_bias(dense_x.weight, true, concat ? out*heads : out) : false
    a = init(ein > 0 ? 3out : 2out, heads)
    negative_slope = convert(Float32, negative_slope)
    GATConv(dense_x, dense_e, b, a, σ, negative_slope, ch, heads, concat, add_self_loops)
end

(l::GATConv)(g::GNNGraph) = GNNGraph(g, ndata=l(g, node_features(g), edge_features(g)))

function (l::GATConv)(g::GNNGraph, x::AbstractMatrix, e::Union{Nothing,AbstractMatrix}=nothing)
    check_num_nodes(g, x)
    @assert !((e === nothing) && (l.dense_e !== nothing)) "Input edge features required for this layer"  
    @assert !((e !== nothing) && (l.dense_e === nothing)) "Input edge features were not specified in the layer constructor"  
    
    if l.add_self_loops
        @assert e === nothing "Using edge features and setting add_self_loops=true at the same time is not yet supported."
        g = add_self_loops(g)
    end
    
    _, chout = l.channel
    heads = l.heads

    Wx = l.dense_x(x)
    Wx = reshape(Wx, chout, heads, :)                   # chout × nheads × nnodes

    m = propagate(message, g, +, l; xi=Wx, xj=Wx, e)                 ## chout × nheads × nnodes
    x = m.β ./ m.α

    if !l.concat
        x = mean(x, dims=2)
    end
    x = reshape(x, :, size(x, 3))  # return a matrix
    x = l.σ.(x .+ l.bias)

    return x
end

function message(l::GATConv, Wxi, Wxj, e)
    _, chout = l.channel
    heads = l.heads

    if e === nothing
        Wxx = vcat(Wxi, Wxj)
    else
        We = l.dense_e(e)
        We = reshape(We, chout, heads, :)                   # chout × nheads × nnodes
        Wxx = vcat(Wxi, Wxj, We)
    end
    aWW = sum(l.a .* Wxx, dims=1)   # 1 × nheads × nedges
    α = exp.(leakyrelu.(aWW, l.negative_slope))       
    return (α = α, β = α .* Wxj)
end

function Base.show(io::IO, l::GATConv)
    (in, ein), out = l.channel
    print(io, "GATConv(", ein == 0 ? in : (in, ein), " => ", out ÷ l.heads)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", negative_slope=", l.negative_slope)
    print(io, ")")
end


@doc raw"""
    GATv2Conv(in => out, [σ; heads, concat, init, bias, negative_slope, add_self_loops])
    GATv2Conv((in, ein) => out, ...)


GATv2 attentional layer from the paper [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491).

Implements the operation
```math
\mathbf{x}_i' = \sum_{j \in N(i) \cup \{i\}} \alpha_{ij} W_1 \mathbf{x}_j
```
where the attention coefficients ``\alpha_{ij}`` are given by
```math
\alpha_{ij} = \frac{1}{z_i} \exp(\mathbf{a}^T LeakyReLU(W_2 \mathbf{x}_i + W_1 \mathbf{x}_j))
```
with ``z_i`` a normalization factor.

In case `ein > 0` is given, edge features of dimension `ein` will be expected in the forward pass 
and the attention coefficients will be calculated as  
```math
\alpha_{ij} = \frac{1}{z_i} \exp(\mathbf{a}^T LeakyReLU(W_3 \mathbf{e}_{j\to i} + W_2 \mathbf{x}_i + W_1 \mathbf{x}_j)).
```

# Arguments

- `in`: The dimension of input node features.
- `ein`: The dimension of input edge features. Default 0 (i.e. no edge features passed in the forward).
- `out`: The dimension of output node features.
- `σ`: Activation function. Default `identity`.
- `bias`: Learn the additive bias if true. Default `true`.
- `heads`: Number attention heads. Default `1`.
- `concat`: Concatenate layer output or not. If not, layer output is averaged over the heads. Default `true`.
- `negative_slope`: The parameter of LeakyReLU.Default `0.2`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `true`.
"""
struct GATv2Conv{T,A1,A2,A3,B,C<:AbstractMatrix,F} <: GNNLayer
    dense_i::A1
    dense_j::A2
    dense_e::A3
    bias::B
    a::C
    σ::F
    negative_slope::T
    channel::Pair{NTuple{2,Int},Int}
    heads::Int
    concat::Bool
    add_self_loops::Bool
end

@functor GATv2Conv
Flux.trainable(l::GATv2Conv) = (l.dense_i, l.dense_j, l.dense_e, l.bias, l.a)

GATv2Conv(ch::Pair{Int,Int}, args...; kws...) = GATv2Conv((ch[1], 0) => ch[2], args...; kws...)

function GATv2Conv(
    ch::Pair{NTuple{2,Int},Int},
    σ=identity;
    heads::Int=1,
    concat::Bool=true,
    negative_slope=0.2,
    init=glorot_uniform,
    bias::Bool=true,
    add_self_loops=true
)
    (in, ein), out = ch

    if add_self_loops
        @assert ein == 0 "Using edge features and setting add_self_loops=true at the same time is not yet supported."
    end
    
    dense_i = Dense(in, out*heads; bias=bias, init=init)
    dense_j = Dense(in, out*heads; bias=false, init=init)
    if ein > 0 
        dense_e = Dense(ein, out*heads; bias=false, init=init)
    else
        dense_e = nothing
    end
    b = bias ? Flux.create_bias(dense_i.weight, true, concat ? out*heads : out) : false
    a = init(out, heads)
    negative_slope = convert(eltype(dense_i.weight), negative_slope)
    GATv2Conv(dense_i, dense_j, dense_e, b, a, σ, negative_slope, ch, heads, concat, add_self_loops)
end

(l::GATv2Conv)(g::GNNGraph) = GNNGraph(g, ndata=l(g, node_features(g), edge_features(g)))

function (l::GATv2Conv)(g::GNNGraph, x::AbstractMatrix, e::Union{Nothing, AbstractMatrix}=nothing)
    check_num_nodes(g, x)
    @assert !((e === nothing) && (l.dense_e !== nothing)) "Input edge features required for this layer"  
    @assert !((e !== nothing) && (l.dense_e === nothing)) "Input edge features were not specified in the layer constructor"  
   
    if l.add_self_loops
        @assert e === nothing "Using edge features and setting add_self_loops=true at the same time is not yet supported."
        g = add_self_loops(g)
    end
    _, out = l.channel
    heads = l.heads

    Wix = reshape(l.dense_i(x), out, heads, :)                                  # out × heads × nnodes
    Wjx = reshape(l.dense_j(x), out, heads, :)                                  # out × heads × nnodes

    m = propagate(message, g, +, l; xi=Wix, xj=Wjx, e)                            # out × heads × nnodes
    x = m.β ./ m.α

    if !l.concat
        x = mean(x, dims=2)
    end
    x = reshape(x, :, size(x, 3))
    x = l.σ.(x .+ l.bias)
    return x  
end

function message(l::GATv2Conv, Wix, Wjx, e)
    _, out = l.channel
    heads = l.heads

    Wx = Wix + Wjx  # Note: this is equivalent to W * vcat(x_i, x_j) as in "How Attentive are Graph Attention Networks?"
    if e !== nothing
        Wx += reshape(l.dense_e(e), out, heads, :)
    end 
    eij = sum(l.a .* leakyrelu.(Wx, l.negative_slope), dims=1)   # 1 × heads × nedges
    α = exp.(eij)
    return (α = α, β = α .* Wjx)
end

function Base.show(io::IO, l::GATv2Conv)
    (in, ein), out = l.channel
    print(io, "GATv2Conv(", ein == 0 ? in : (in, ein), " => ", out ÷ l.heads)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", negative_slope=", l.negative_slope)
    print(io, ")")
end


@doc raw"""
    GatedGraphConv(out, num_layers; aggr=+, init=glorot_uniform)

Gated graph convolution layer from [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493).

Implements the recursion
```math
\begin{aligned}
\mathbf{h}^{(0)}_i &= [\mathbf{x}_i; \mathbf{0}] \\
\mathbf{h}^{(l)}_i &= GRU(\mathbf{h}^{(l-1)}_i, \square_{j \in N(i)} W \mathbf{h}^{(l-1)}_j)
\end{aligned}
```

where ``\mathbf{h}^{(l)}_i`` denotes the ``l``-th hidden variables passing through GRU. The dimension of input ``\mathbf{x}_i`` needs to be less or equal to `out`.

# Arguments

- `out`: The dimension of output features.
- `num_layers`: The number of gated recurrent unit.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `init`: Weight initialization function.
"""
struct GatedGraphConv{W<:AbstractArray{<:Number,3},R,A} <: GNNLayer
    weight::W
    gru::R
    out_ch::Int
    num_layers::Int
    aggr::A
end

@functor GatedGraphConv

function GatedGraphConv(out_ch::Int, num_layers::Int;
                        aggr=+, init=glorot_uniform)
    w = init(out_ch, out_ch, num_layers)
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(w, gru, out_ch, num_layers, aggr)
end

# remove after https://github.com/JuliaDiff/ChainRules.jl/pull/521
@non_differentiable fill!(x...)

function (l::GatedGraphConv)(g::GNNGraph, H::AbstractMatrix{S}) where {S<:Real}
    check_num_nodes(g, H)
    m, n = size(H)
    @assert (m <= l.out_ch) "number of input features must less or equals to output features."
    if m < l.out_ch
        Hpad = similar(H, S, l.out_ch - m, n)
        H = vcat(H, fill!(Hpad, 0))
    end
    for i = 1:l.num_layers
        M = view(l.weight, :, :, i) * H
        M = propagate(copy_xj, g, l.aggr; xj=M)
        H, _ = l.gru(H, M)
    end
    H
end

function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv(($(l.out_ch) => $(l.out_ch))^$(l.num_layers)")
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


@doc raw"""
    EdgeConv(nn; aggr=max)

Edge convolutional layer from paper [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829).

Performs the operation
```math
\mathbf{x}_i' = \square_{j \in N(i)}\, nn([\mathbf{x}_i; \mathbf{x}_j - \mathbf{x}_i])
```

where `nn` generally denotes a learnable function, e.g. a linear layer or a multi-layer perceptron.

# Arguments

- `nn`: A (possibly learnable) function. 
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
"""
struct EdgeConv{NN,A} <: GNNLayer
    nn::NN
    aggr::A
end

@functor EdgeConv

EdgeConv(nn; aggr=max) = EdgeConv(nn, aggr)

function (l::EdgeConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    message(l, xi, xj, e) = l.nn(vcat(xi, xj .- xi))
    x = propagate(message, g, l.aggr, l, xi=x, xj=x)
    return x
end

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


@doc raw"""
    GINConv(f, ϵ; aggr=+)

Graph Isomorphism convolutional layer from paper [How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826.pdf).

Implements the graph convolution
```math
\mathbf{x}_i' = f_\Theta\left((1 + \epsilon) \mathbf{x}_i + \sum_{j \in N(i)} \mathbf{x}_j \right)
```
where ``f_\Theta`` typically denotes a learnable function, e.g. a linear layer or a multi-layer perceptron.

# Arguments

- `f`: A (possibly learnable) function acting on node features. 
- `ϵ`: Weighting factor.
"""
struct GINConv{R<:Real,NN,A} <: GNNLayer
    nn::NN
    ϵ::R
    aggr::A
end

@functor GINConv
Flux.trainable(l::GINConv) = (l.nn,)

GINConv(nn, ϵ; aggr=+) = GINConv(nn, ϵ, aggr)

function (l::GINConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    m = propagate(copy_xj, g, l.aggr, xj=x)
    l.nn((1 + ofeltype(x, l.ϵ)) * x + m)
end

function Base.show(io::IO, l::GINConv)
    print(io, "GINConv($(l.nn)")
    print(io, ", $(l.ϵ)")
    print(io, ")")
end


@doc raw"""
    NNConv(in => out, f, σ=identity; aggr=+, bias=true, init=glorot_uniform)

The continuous kernel-based convolutional operator from the 
[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) paper. 
This convolution is also known as the edge-conditioned convolution from the 
[Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) paper.

Performs the operation

```math
\mathbf{x}_i' = W \mathbf{x}_i + \square_{j \in N(i)} f_\Theta(\mathbf{e}_{j\to i})\,\mathbf{x}_j
```

where ``f_\Theta``  denotes a learnable function (e.g. a linear layer or a multi-layer perceptron).
Given an input of batched edge features `e` of size `(num_edge_features, num_edges)`, 
the function `f` will return an batched matrices array whose size is `(out, in, num_edges)`.
For convenience, also functions returning a single `(out*in, num_edges)` matrix are allowed.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `f`: A (possibly learnable) function acting on edge features.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `σ`: Activation function.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct NNConv{W,B,NN,F,A} <: GNNLayer
    weight::W
    bias::B
    nn::NN
    σ::F
    aggr::A
end

@functor NNConv

function NNConv(ch::Pair{Int,Int}, nn, σ=identity; aggr=+, bias=true, init=glorot_uniform)
    in, out = ch
    W = init(out, in)
    b = bias ? Flux.create_bias(W, true, out) : false
    return NNConv(W, b, nn, σ, aggr)
end

function (l::NNConv)(g::GNNGraph, x::AbstractMatrix, e)
    check_num_nodes(g, x)    

    m = propagate(message, g, l.aggr, l, xj=x, e=e)
    return l.σ.(l.weight*x .+ m .+ l.bias)
end

function message(l::NNConv, xi, xj, e)
    nin, nedges = size(xj)
    W = reshape(l.nn(e), (:, nin, nedges))
    xj = reshape(xj, (nin, 1, nedges)) # needed by batched_mul
    m = NNlib.batched_mul(W, xj)
    return reshape(m, :, nedges)
end

(l::NNConv)(g::GNNGraph) = GNNGraph(g, ndata=l(g, node_features(g), edge_features(g)))

function Base.show(io::IO, l::NNConv)
    out, in = size(l.weight)
    print(io, "NNConv($in => $out")
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


@doc raw"""
    SAGEConv(in => out, σ=identity; aggr=mean, bias=true, init=glorot_uniform)

GraphSAGE convolution layer from paper [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf).

Performs:
```math
\mathbf{x}_i' = W \cdot [\mathbf{x}_i; \square_{j \in \mathcal{N}(i)} \mathbf{x}_j]
```

where the aggregation type is selected by `aggr`.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
"""
struct SAGEConv{W<:AbstractMatrix, B, F, A} <: GNNLayer
    weight::W
    bias::B
    σ::F
    aggr::A
end

@functor SAGEConv

function SAGEConv(ch::Pair{Int,Int}, σ=identity; aggr=mean,
                   init=glorot_uniform, bias::Bool=true)
    in, out = ch
    W = init(out, 2*in)
    b = bias ? Flux.create_bias(W, true, out) : false
    SAGEConv(W, b, σ, aggr)
end

function (l::SAGEConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    m = propagate(copy_xj, g, l.aggr, xj=x)
    x = l.σ.(l.weight * vcat(x, m) .+ l.bias)
    return x 
end

function Base.show(io::IO, l::SAGEConv)
    out_channel, in_channel = size(l.weight)
    print(io, "SAGEConv(", in_channel ÷ 2, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


@doc raw"""
    ResGatedGraphConv(in => out, act=identity; init=glorot_uniform, bias=true)

The residual gated graph convolutional operator from the [Residual Gated Graph ConvNets](https://arxiv.org/abs/1711.07553) paper.

The layer's forward pass is given by

```math
\mathbf{x}_i' = act\big(U\mathbf{x}_i + \sum_{j \in N(i)} \eta_{ij} V \mathbf{x}_j\big),
```
where the edge gates ``\eta_{ij}`` are given by

```math
\eta_{ij} = sigmoid(A\mathbf{x}_i + B\mathbf{x}_j).
```

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `act`: Activation function.
- `init`: Weight matrices' initializing function. 
- `bias`: Learn an additive bias if true.
"""
struct ResGatedGraphConv{W,B,F} <: GNNLayer
    A::W
    B::W
    U::W
    V::W
    bias::B
    σ::F
end

@functor ResGatedGraphConv

function ResGatedGraphConv(ch::Pair{Int,Int}, σ=identity;
                      init=glorot_uniform, bias::Bool=true)
    in, out = ch             
    A = init(out, in)
    B = init(out, in)
    U = init(out, in)
    V = init(out, in)
    b = bias ? Flux.create_bias(A, true, out) : false
    return ResGatedGraphConv(A, B, U, V, b, σ)
end

function (l::ResGatedGraphConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)

    message(xi, xj, e) = sigmoid.(xi.Ax .+ xj.Bx) .* xj.Vx
    
    Ax = l.A * x
    Bx = l.B * x
    Vx = l.V * x
    
    m = propagate(message, g, +, xi=(; Ax), xj=(; Bx, Vx))
    
    return l.σ.(l.U*x .+ m .+ l.bias)                                      
end

function Base.show(io::IO, l::ResGatedGraphConv)
    out_channel, in_channel = size(l.A)
    print(io, "ResGatedGraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end



@doc raw"""
    CGConv((in, ein) => out, f, act=identity; bias=true, init=glorot_uniform, residual=false)
    CGConv(in => out, ...)

The crystal graph convolutional layer from the paper
[Crystal Graph Convolutional Neural Networks for an Accurate and
Interpretable Prediction of Material Properties](https://arxiv.org/pdf/1710.10324.pdf).
Performs the operation

```math
\mathbf{x}_i' = \mathbf{x}_i + \sum_{j\in N(i)}\sigma(W_f \mathbf{z}_{ij} + \mathbf{b}_f)\, act(W_s \mathbf{z}_{ij} + \mathbf{b}_s)
```

where ``\mathbf{z}_{ij}``  is the node and edge features concatenation 
``[\mathbf{x}_i; \mathbf{x}_j; \mathbf{e}_{j\to i}]`` 
and ``\sigma`` is the sigmoid function.
The residual ``\mathbf{x}_i`` is added only if `residual=true` and the output size is the same 
as the input size.

# Arguments

- `in`: The dimension of input node features.
- `ein`: The dimension of input edge features. 
If `ein` is not given, assumes that no edge features are passed as input in the forward pass.
- `out`: The dimension of output node features.
- `act`: Activation function.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
- `residual`: Add a residual connection.

# Examples 

```julia
g = rand_graph(5, 6)
x = rand(Float32, 2, g.num_nodes)
e = rand(Float32, 3, g.num_edges)

l = CGConv((2, 3) => 4, tanh)
y = l(g, x, e)    # size: (4, num_nodes)

# No edge features
l = CGConv(2 => 4, tanh)
y = l(g, x)    # size: (4, num_nodes)
```
"""
struct CGConv{D1,D2} <: GNNLayer
    ch::Pair{NTuple{2,Int},Int}
    dense_f::D1
    dense_s::D2
    residual::Bool
end

@functor CGConv

CGConv(ch::Pair{Int,Int}, args...; kws...) = CGConv((ch[1], 0) => ch[2], args...; kws...)

function CGConv(ch::Pair{NTuple{2,Int},Int}, act=identity; residual=false, bias=true, init=glorot_uniform)
    (nin, ein), out = ch
    dense_f = Dense(2nin+ein, out, sigmoid; bias, init)
    dense_s = Dense(2nin+ein, out, act; bias, init)
    return CGConv(ch, dense_f, dense_s, residual)
end

function (l::CGConv)(g::GNNGraph, x::AbstractMatrix, e::Union{Nothing, AbstractMatrix}=nothing)
    check_num_nodes(g, x)
    if e !== nothing
        check_num_edges(g, e)
    end

    m = propagate(message, g, +, l, xi=x, xj=x, e=e)

    if l.residual
        if size(x, 1) == size(m, 1)
            m += x
        else
            @warn "number of output features different from number of input features, residual not applied."
        end
    end

    return m
end

function message(l::CGConv, xi, xj, e)
    if e !== nothing
        z = vcat(xi, xj, e)
    else
        z = vcat(xi, xj)
    end
    return l.dense_f(z) .* l.dense_s(z)
end

(l::CGConv)(g::GNNGraph) = GNNGraph(g, ndata=l(g, node_features(g), edge_features(g)))

function Base.show(io::IO, l::CGConv)
    print(io, "CGConv($(l.ch)")
    l.dense_s.σ == identity || print(io, ", ", l.dense_s.σ)
    print(io, ", residual=$(l.residual)")
    print(io, ")")
end


@doc raw"""
    AGNNConv(init_beta=1f0)

Attention-based Graph Neural Network layer from paper [Attention-based
Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735).

The forward pass is given by
```math
\mathbf{x}_i' = \sum_{j \in {N(i) \cup \{i\}}} \alpha_{ij} W \mathbf{x}_j
```
where the attention coefficients ``\alpha_{ij}`` are given by
```math
\alpha_{ij} =\frac{e^{\beta \cos(\mathbf{x}_i, \mathbf{x}_j)}}
                  {\sum_{j'}e^{\beta \cos(\mathbf{x}_i, \mathbf{x}_{j'})}}
```
with the cosine distance defined by
```math 
\cos(\mathbf{x}_i, \mathbf{x}_j) = 
  \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\lVert\mathbf{x}_i\rVert \lVert\mathbf{x}_j\rVert}
```
and ``\beta`` a trainable parameter.

# Arguments

- `init_beta`: The initial value of ``\beta``.
"""
struct AGNNConv{A<:AbstractVector} <: GNNLayer
    β::A
end

@functor AGNNConv

function AGNNConv(init_beta = 1f0)
    AGNNConv([init_beta])
end

function (l::AGNNConv)(g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    g = add_self_loops(g)

    xn = x ./ sqrt.(sum(x.^2, dims=1))
    cos_dist = apply_edges(xi_dot_xj, g, xi=xn, xj=xn)
    α = softmax_edge_neighbors(g, l.β .* cos_dist)

    x = propagate(g, +; xj=x, e=α) do xi, xj, α
            α .* xj
        end

    return x  
end

@doc raw"""
    MEGNetConv(ϕe, ϕv; aggr=mean)
    MEGNetConv(in => out; aggr=mean)

Convolution from [Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://arxiv.org/pdf/1812.05055.pdf)
paper. In the forward pass, takes as inputs node features `x` and edge features `e` and returns
updated features `x'` and `e'` according to 

```math
\begin{aligned}
\mathbf{e}_{i\to j}'  = \phi_e([\mathbf{x}_i;\,  \mathbf{x}_j;\,  \mathbf{e}_{i\to j}]),\\
\mathbf{x}_{i}'  = \phi_v([\mathbf{x}_i;\, \square_{j\in \mathcal{N}(i)}\,\mathbf{e}_{j\to i}']).
\end{aligned}
```

`aggr` defines the aggregation to be performed.

If the neural networks `ϕe` and  `ϕv` are not provided, they will be constructed from
the `in` and `out` arguments instead as multi-layer perceptron with one hidden layer and `relu` 
activations.

# Examples

```julia
g = rand_graph(10, 30)
x = randn(3, 10)
e = randn(3, 30)
m = MEGNetConv(3 => 3)
x′, e′ = m(g, x, e)
```
"""
struct MEGNetConv{TE,TV,A} <: GNNLayer
    ϕe::TE
    ϕv::TV
    aggr::A
end

@functor MEGNetConv

MEGNetConv(ϕe, ϕv; aggr=mean) = MEGNetConv(ϕe, ϕv, aggr)

function MEGNetConv(ch::Pair{Int,Int}; aggr=mean)
    nin, nout = ch 
    ϕe = Chain(Dense(3nin, nout, relu),
               Dense(nout, nout))

    ϕv = Chain(Dense(nin + nout, nout, relu),
               Dense(nout, nout))

    MEGNetConv(ϕe, ϕv; aggr)
end

function (l::MEGNetConv)(g::GNNGraph)
    x, e = l(g, node_features(g), edge_features(g))
    g = GNNGraph(g, ndata=x, edata=e)
end

function (l::MEGNetConv)(g::GNNGraph, x::AbstractMatrix, e::AbstractMatrix)
    check_num_nodes(g, x)

    ē = apply_edges(g, xi=x, xj=x, e=e) do xi, xj, e
        l.ϕe(vcat(xi, xj, e))
    end

    xᵉ = aggregate_neighbors(g, l.aggr, ē)

    x̄ = l.ϕv(vcat(x, xᵉ))

    return x̄, ē
end

@doc raw"""
    GMMConv((in, ein) => out, σ=identity; K=1, bias=true, init=glorot_uniform, residual=false)

Graph mixture model convolution layer from the paper [Geometric deep learning on graphs and manifolds using mixture model CNNs](https://arxiv.org/abs/1611.08402)
Performs the operation
```math
\mathbf{x}_i' = \mathbf{x}_i + \frac{1}{|N(i)|} \sum_{j\in N(i)}\frac{1}{K}\sum_{k=1}^K \mathbf{w}_k(\mathbf{e}_{j\to i}) \odot \Theta_k \mathbf{x}_j
```
where ``w^a_{k}(e^a)`` for feature `a` and kernel `k` is given by
```math
w^a_{k}(e^a) = \exp(-\frac{1}{2}(e^a - \mu^a_k)^T (\Sigma^{-1})^a_k(e^a - \mu^a_k))
```
``\Theta_k, \mu^a_k, (\Sigma^{-1})^a_k`` are learnable parameters.

The input to the layer is a node feature array `x` of size `(num_features, num_nodes)` and
edge pseudo-coordinate array `e` of size `(num_features, num_edges)`
The residual ``\mathbf{x}_i`` is added only if `residual=true` and the output size is the same 
as the input size.

# Arguments 

- `in`: Number of input node features.
- `ein`: Number of input edge features.
- `out`: Number of output features.
- `σ`: Activation function. Default `identity`.
- `K`: Number of kernels. Default `1`.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `residual`: Residual conncetion. Default `false`.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s,t)
nin, ein, out, K = 4, 10, 7, 8 
x = randn(Float32, nin, g.num_nodes)
e = randn(Float32, ein, g.num_edges)

# create layer
l = GMMConv((nin, ein) => out, K=K)

# forward pass
l(g, x, e)
```
"""
struct GMMConv{A<:AbstractMatrix, B, F} <:GNNLayer
    mu::A
    sigma_inv::A
    bias::B
    σ::F
    ch::Pair{NTuple{2,Int},Int}
    K::Int
    dense_x::Dense
    residual::Bool
end

@functor GMMConv

function GMMConv(ch::Pair{NTuple{2,Int},Int}, 
                σ=identity;
                K::Int=1,
                bias::Bool=true,
                init=Flux.glorot_uniform,
                residual=false)
    
    (nin, ein), out = ch
    mu = init(ein, K)
    sigma_inv = init(ein, K)
    b = bias ? Flux.create_bias(mu, true, out) : false
    dense_x = Dense(nin, out*K, bias=false)
    GMMConv(mu, sigma_inv, b, σ, ch, K, dense_x, residual)
end

function (l::GMMConv)(g::GNNGraph, x::AbstractMatrix, e::AbstractMatrix)
    (nin, ein), out = l.ch #Notational Simplicity

    @assert (ein == size(e)[1] && g.num_edges == size(e)[2]) "Pseudo-cordinate dimension is not equal to (ein,num_edge)"

    num_edges = g.num_edges
    w = reshape(e, (ein, 1, num_edges))
    mu = reshape(l.mu, (ein, l.K, 1))
    
    w = @. ((w - mu)^2) / 2
    w = w .* reshape(l.sigma_inv.^2, (ein, l.K, 1))
    w = exp.(sum(w, dims = 1 )) # (1, K, num_edge) 

    xj = reshape(l.dense_x(x), (out, l.K, :)) # (out, K, num_nodes) 

    m = propagate(e_mul_xj, g, mean, xj=xj, e=w)
    m = dropdims(mean(m, dims=2), dims=2) # (out, num_nodes)  

    m = l.σ(m .+ l.bias)
    
    if l.residual
        if size(x, 1) == size(m, 1)
            m += x
        else
            @warn "Residual not applied : output feature is not equal to input_feature"
        end
    end

    return m
end
                            
(l::GMMConv)(g::GNNGraph) = GNNGraph(g, ndata=l(g, node_features(g), edge_features(g)))

function Base.show(io::IO, l::GMMConv)
    (nin, ein), out = l.ch
    print(io, "GMMConv((", nin, ",", ein, ")=>", out)
    l.σ == identity || print(io, ", σ=", l.dense_s.σ)
    print(io, ", K=", l.K)
    l.residual==true || print(io, ", residual=", l.residual)
    print(io, ")")
end

@doc raw"""
    SGConv(int => out, k=1; [bias, init, add_self_loops, use_edge_weight])
                                
SGC layer from [Simplifying Graph Convolutional Networks](https://arxiv.org/pdf/1902.07153.pdf)
Performs operation
```math
H^{K} = (\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2})^K X \Theta
```
where ``\tilde{A}`` is ``A + I``.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `k` : Number of hops k. Default `1`.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. Default `false`.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(3, g.num_nodes)

# create layer
l = SGConv(3 => 5; add_self_loops = true) 

# forward pass
y = l(g, x)       # size:  5 × num_nodes

# convolution with edge weights
w = [1.1, 0.1, 2.3, 0.5]
y = l(g, x, w)

# Edge weights can also be embedded in the graph.
g = GNNGraph(s, t, w)
l = SGConv(3 => 5, add_self_loops = true, use_edge_weight=true) 
y = l(g, x) # same as l(g, x, w) 
```
"""
struct SGConv{A<:AbstractMatrix, B} <: GNNLayer
    weight::A
    bias::B
    k::Int
    add_self_loops::Bool
    use_edge_weight::Bool
end

@functor SGConv

function SGConv(ch::Pair{Int,Int}, k=1;
                 init=glorot_uniform,
                 bias::Bool=true,
                 add_self_loops=true,
                 use_edge_weight=false)
    in, out = ch
    W = init(out, in)
    b = bias ? Flux.create_bias(W, true, out) : false
    SGConv(W, b, k, add_self_loops, use_edge_weight)
end

function (l::SGConv)(g::GNNGraph, x::AbstractMatrix{T}, edge_weight::EW=nothing) where
    {T, EW<:Union{Nothing,AbstractVector}}     
    @assert !(g isa GNNGraph{<:ADJMAT_T} && edge_weight !== nothing) "Providing external edge_weight is not yet supported for adjacency matrix graphs"

    if edge_weight !== nothing
        @assert length(edge_weight) == g.num_edges "Wrong number of edge weights (expected $(g.num_edges) but given $(length(edge_weight)))"
    end

    if l.add_self_loops
        g = add_self_loops(g)
        if edge_weight !== nothing
            edge_weight = [edge_weight; fill!(similar(edge_weight, g.num_nodes), 1)]
            @assert length(edge_weight) == g.num_edges
        end
    end
    Dout, Din = size(l.weight)
    if Dout < Din
        x = l.weight * x
    end
    d = degree(g, T; dir=:in, edge_weight)
    c = 1 ./ sqrt.(d)
    for iter in 1:l.k
        x = x .* c'
        if edge_weight !== nothing            
            x = propagate(e_mul_xj, g, +, xj=x, e=edge_weight)
        elseif l.use_edge_weight        
            x = propagate(w_mul_xj, g, +, xj=x)
        else
            x = propagate(copy_xj, g, +, xj=x)
        end
        x = x .* c'
    end    
    if Dout >= Din
        x = l.weight * x
    end
    return (x .+ l.bias)
end

function (l::SGConv)(g::GNNGraph{<:ADJMAT_T}, x::AbstractMatrix, edge_weight::AbstractVector)
    g = GNNGraph(edge_index(g)...; g.num_nodes)
    return l(g, x, edge_weight)
end

function Base.show(io::IO, l::SGConv)
    out, in = size(l.weight)
    print(io, "SGConv($in => $out")
    l.k == 1 || print(io, ", ", l.k)
    print(io, ")")
end



@doc raw"""
    EdgeConv((in, ein) => out; hidden_size=2in, residual=false)
    EdgeConv(in => out; hidden_size=2in, residual=false)

Equivariant Graph Convolutional Layer from [E(n) Equivariant Graph
Neural Networks](https://arxiv.org/abs/2102.09844).

The layer performs the following operation:

```math
\begin{aligned}
\mathbf{m}_{j\to i} &=\phi_e(\mathbf{h}_i, \mathbf{h}_j, \lVert\mathbf{x}_i-\mathbf{x}_j\rVert^2, \mathbf{e}_{j\to i}),\\
\mathbf{x}_i' &= \mathbf{x}_i + C_i\sum_{j\in\mathcal{N}(i)}(\mathbf{x}_i-\mathbf{x}_j)\phi_x(\mathbf{m}_{j\to i}),\\
\mathbf{m}_i &= C_i\sum_{j\in\mathcal{N}(i)} \mathbf{m}_{j\to i},\\
\mathbf{h}_i' &= \mathbf{h}_i + \phi_h(\mathbf{h}_i, \mathbf{m}_i)
\end{aligned}
```
where ``\mathbf{h}_i``, ``\mathbf{x}_i``, ``\mathbf{e}_{j\to i}`` are invariant node features, equivariant node
features, and edge features respectively. ``\phi_e``, ``\phi_h``, and
``\phi_x`` are two-layer MLPs. `C` is a constant for normalization,
computed as ``1/|\mathcal{N}(i)|``.


# Constructor Arguments

- `in`: Number of input features for `h`.
- `out`: Number of output features for `h`.
- `ein`: Number of input edge features.
- `hidden_size`: Hidden representation size.
- `residual`: If `true`, add a residual connection. Only possible if `in == out`. Default `false`.

# Forward Pass 

    l(g, x, h, e=nothing)
                     
## Forward Pass Arguments:

- `g` : The graph.
- `x` : Matrix of equivariant node coordinates.
- `h` : Matrix of invariant node features.
- `e` : Matrix of invariant edge features. Default `nothing`.

Returns updated `h` and `x`.

# Examples

```julia
g = rand_graph(10, 10)
h = randn(Float32, 5, g.num_nodes)
x = randn(Float32, 3, g.num_nodes)
egnn = EGNNConv(5 => 6, 10)
hnew, xnew = egnn(g, h, x)
```
"""
struct EGNNConv{TE,TX,TH,NF}  <: GNNLayer
    ϕe::TE
    ϕx::TX
    ϕh::TH
    num_features::NF
    residual::Bool
end

@functor EGNNConv
        
EGNNConv(ch::Pair{Int,Int}, hidden_size=2*ch[1]; residual=false) = EGNNConv((ch[1], 0) => ch[2]; hidden_size, residual) 

#Follows reference implementation at https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py
function EGNNConv(ch::Pair{NTuple{2, Int}, Int}; hidden_size::Int=2*ch[1][1], residual=false)
    (in_size, edge_feat_size), out_size = ch
    act_fn = swish

    # +1 for the radial feature: ||x_i - x_j||^2
    ϕe = Chain(Dense(in_size * 2 + edge_feat_size + 1 => hidden_size, act_fn),
               Dense(hidden_size => hidden_size, act_fn))

    ϕh = Chain(Dense(in_size + hidden_size, hidden_size, swish),
               Dense(hidden_size, out_size))

    ϕx = Chain(Dense(hidden_size, hidden_size, swish),
               Dense(hidden_size, 1, bias=false))

    num_features = (in=in_size, edge=edge_feat_size, out=out_size, hidden=hidden_size)
    if residual 
        @assert in_size == out_size "Residual connection only possible if in_size == out_size"
    end
    return EGNNConv(ϕe, ϕx, ϕh, num_features, residual)
end

function (l::EGNNConv)(g::GNNGraph, h::AbstractMatrix, x::AbstractMatrix, e=nothing)
    if l.num_features.edge > 0
        @assert e !== nothing "Edge features must be provided."
    end
    @assert size(h, 1) == l.num_features.in "Input features must match layer input size."

    x_diff = apply_edges(xi_sub_xj, g, x, x)
    sqnorm_xdiff = sum(x_diff .^ 2, dims=1)
    x_diff = x_diff ./ (sqrt.(sqnorm_xdiff) .+ 1f-6)

    msg = apply_edges(message, g, l, 
            xi=(; h), xj=(; h), e=(; e, x_diff, sqnorm_xdiff))
    h_aggr = aggregate_neighbors(g, +, msg.h)
    x_aggr = aggregate_neighbors(g, mean, msg.x)
    
    hnew = l.ϕh(vcat(h, h_aggr))
    if l.residual
        h = h .+ hnew
    else
        h = hnew
    end
    x = x .+ x_aggr
    return h, x
end

function message(l::EGNNConv, xi, xj, e)
    if l.num_features.edge > 0
        f = vcat(xi.h, xj.h, e.sqnorm_xdiff, e.e)
    else
        f = vcat(xi.h, xj.h, e.sqnorm_xdiff)
    end

    msg_h = l.ϕe(f)
    msg_x = l.ϕx(msg_h) .* e.x_diff
    return (; x=msg_x, h=msg_h)
end

function Base.show(io::IO, l::EGNNConv)
    ne = l.num_features.edge 
    nin = l.num_features.in
    nout = l.num_features.out
    nh = l.num_features.hidden
    print(io, "EGNNConv(($nin, $ne) => $nout; hidden_size=$nh")
    if l.residual
        print(io, ", residual=true")
    end
    print(io, ")")
end
