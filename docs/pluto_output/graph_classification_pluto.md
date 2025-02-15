```@raw html
<style>
    table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    pre, div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }

    .admonition-body {
        padding: 0em 1.25em !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "a5e4a6292222b003f3138c076d1f0d4b2c340e5b730135690d834f42a3edb316"
    julia_version = "1.8.2"
-->







<div class="markdown"><p><em>This Pluto notebook is a julia adaptation of the Pytorch Geometric tutorials that can be found <a href="https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html">here</a>.</em></p><p>In this tutorial session we will have a closer look at how to apply <strong>Graph Neural Networks (GNNs) to the task of graph classification</strong>. Graph classification refers to the problem of classifying entire graphs (in contrast to nodes), given a <strong>dataset of graphs</strong>, based on some structural graph properties. Here, we want to embed entire graphs, and we want to embed those graphs in such a way so that they are linearly separable given a task at hand.</p><p>The most common task for graph classification is <strong>molecular property prediction</strong>, in which molecules are represented as graphs, and the task may be to infer whether a molecule inhibits HIV virus replication or not.</p><p>The TU Dortmund University has collected a wide range of different graph classification datasets, known as the <a href="https://chrsmrrs.github.io/datasets/"><strong>TUDatasets</strong></a>, which are also accessible via MLDatasets.jl. Let's load and inspect one of the smaller ones, the <strong>MUTAG dataset</strong>:</p></div>

<pre class='language-julia'><code class='language-julia'>dataset = TUDataset("MUTAG")</code></pre>
<pre class="code-output documenter-example-output" id="var-dataset">dataset TUDataset:
  name        =&gt;    MUTAG
  metadata    =&gt;    Dict{String, Any} with 1 entry
  graphs      =&gt;    188-element Vector{MLDatasets.Graph}
  graph_data  =&gt;    (targets = "188-element Vector{Int64}",)
  num_nodes   =&gt;    3371
  num_edges   =&gt;    7442
  num_graphs  =&gt;    188</pre>

<pre class='language-julia'><code class='language-julia'>dataset.graph_data.targets |&gt; union</code></pre>
<pre class="code-output documenter-example-output" id="var-hash194771">2-element Vector{Int64}:
  1
 -1</pre>

<pre class='language-julia'><code class='language-julia'>g1, y1  = dataset[1] #get the first graph and target</code></pre>
<pre class="code-output documenter-example-output" id="var-y1">(graphs = Graph(17, 38), targets = 1)</pre>

<pre class='language-julia'><code class='language-julia'>reduce(vcat, g.node_data.targets for (g,_) in dataset) |&gt; union</code></pre>
<pre class="code-output documenter-example-output" id="var-hash163982">7-element Vector{Int64}:
 0
 1
 2
 3
 4
 5
 6</pre>

<pre class='language-julia'><code class='language-julia'>reduce(vcat, g.edge_data.targets for (g,_) in dataset)|&gt; union</code></pre>
<pre class="code-output documenter-example-output" id="var-hash766940">4-element Vector{Int64}:
 0
 1
 2
 3</pre>


<div class="markdown"><p>This dataset provides <strong>188 different graphs</strong>, and the task is to classify each graph into <strong>one out of two classes</strong>.</p><p>By inspecting the first graph object of the dataset, we can see that it comes with <strong>17 nodes</strong> and <strong>38 edges</strong>. It also comes with exactly <strong>one graph label</strong>, and provides additional node labels (7 classes) and edge labels (4 classes). However, for the sake of simplicity, we will not make use of edge labels.</p></div>


<div class="markdown"><p>We now convert the MLDatasets.jl graph types to our <code>GNNGraph</code>s and we also onehot encode both the node labels (which will be used as input features) and the graph labels (what we want to predict):  </p></div>

<pre class='language-julia'><code class='language-julia'>begin
    graphs = mldataset2gnngraph(dataset)
    graphs = [GNNGraph(g, 
                       ndata=Float32.(onehotbatch(g.ndata.targets, 0:6)),
                       edata=nothing) 
              for g in graphs]
    y = onehotbatch(dataset.graph_data.targets, [-1, 1])
end</code></pre>
<pre class="code-output documenter-example-output" id="var-y">2×188 OneHotMatrix(::Vector{UInt32}) with eltype Bool:
 ⋅  1  1  ⋅  1  ⋅  1  ⋅  1  ⋅  ⋅  ⋅  ⋅  1  …  ⋅  ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅  1  1  ⋅  1
 1  ⋅  ⋅  1  ⋅  1  ⋅  1  ⋅  1  1  1  1  ⋅     1  1  1  ⋅  1  ⋅  ⋅  1  1  ⋅  ⋅  1  ⋅</pre>


<div class="markdown"><p>We have some useful utilities for working with graph datasets, <em>e.g.</em>, we can shuffle the dataset and use the first 150 graphs as training graphs, while using the remaining ones for testing:</p></div>

<pre class='language-julia'><code class='language-julia'>train_data, test_data = splitobs((graphs, y), at=150, shuffle=true) |&gt; getobs</code></pre>
<pre class="code-output documenter-example-output" id="var-train_data">((GraphNeuralNetworks.GNNGraphs.GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}[GNNGraph(25, 58), GNNGraph(16, 34), GNNGraph(23, 54), GNNGraph(20, 46), GNNGraph(18, 40), GNNGraph(13, 28), GNNGraph(12, 24), GNNGraph(16, 34), GNNGraph(19, 44), GNNGraph(22, 50)  …  GNNGraph(11, 22), GNNGraph(23, 54), GNNGraph(17, 38), GNNGraph(13, 28), GNNGraph(16, 34), GNNGraph(12, 26), GNNGraph(20, 44), GNNGraph(22, 50), GNNGraph(19, 44), GNNGraph(20, 44)], Bool[0 0 … 0 0; 1 1 … 1 1]), (GraphNeuralNetworks.GNNGraphs.GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}[GNNGraph(16, 34), GNNGraph(12, 26), GNNGraph(24, 50), GNNGraph(21, 44), GNNGraph(22, 50), GNNGraph(23, 54), GNNGraph(24, 50), GNNGraph(15, 34), GNNGraph(21, 44), GNNGraph(23, 54)  …  GNNGraph(23, 50), GNNGraph(25, 56), GNNGraph(23, 54), GNNGraph(23, 50), GNNGraph(11, 22), GNNGraph(22, 50), GNNGraph(23, 54), GNNGraph(24, 50), GNNGraph(20, 46), GNNGraph(13, 28)], Bool[1 1 … 0 1; 0 0 … 1 0]))</pre>

<pre class='language-julia'><code class='language-julia'>begin
    train_loader = DataLoader(train_data, batchsize=64, shuffle=true)
    test_loader = DataLoader(test_data, batchsize=64, shuffle=false)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-test_loader">DataLoader{Tuple{Vector{GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}}, OneHotArrays.OneHotMatrix{UInt32, Vector{UInt32}}}, Random._GLOBAL_RNG, Val{nothing}}((GraphNeuralNetworks.GNNGraphs.GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}[GNNGraph(16, 34), GNNGraph(12, 26), GNNGraph(24, 50), GNNGraph(21, 44), GNNGraph(22, 50), GNNGraph(23, 54), GNNGraph(24, 50), GNNGraph(15, 34), GNNGraph(21, 44), GNNGraph(23, 54)  …  GNNGraph(23, 50), GNNGraph(25, 56), GNNGraph(23, 54), GNNGraph(23, 50), GNNGraph(11, 22), GNNGraph(22, 50), GNNGraph(23, 54), GNNGraph(24, 50), GNNGraph(20, 46), GNNGraph(13, 28)], Bool[1 1 … 0 1; 0 0 … 1 0]), 64, false, true, false, false, Val{nothing}(), Random._GLOBAL_RNG())</pre>


<div class="markdown"><p>Here, we opt for a <code>batch_size</code> of 64, leading to 3 (randomly shuffled) mini-batches, containing all <span class="tex">$2 \cdot 64+22 = 150$</span> graphs.</p></div>


```
## Mini-batching of graphs
```@raw html
<div class="markdown">
<p>Since graphs in graph classification datasets are usually small, a good idea is to <strong>batch the graphs</strong> before inputting them into a Graph Neural Network to guarantee full GPU utilization. In the image or language domain, this procedure is typically achieved by <strong>rescaling</strong> or <strong>padding</strong> each example into a set of equally-sized shapes, and examples are then grouped in an additional dimension. The length of this dimension is then equal to the number of examples grouped in a mini-batch and is typically referred to as the <code>batchsize</code>.</p><p>However, for GNNs the two approaches described above are either not feasible or may result in a lot of unnecessary memory consumption. Therefore, GraphNeuralNetworks.jl opts for another approach to achieve parallelization across a number of examples. Here, adjacency matrices are stacked in a diagonal fashion (creating a giant graph that holds multiple isolated subgraphs), and node and target features are simply concatenated in the node dimension (the last dimension).</p><p>This procedure has some crucial advantages over other batching procedures:</p><ol><li><p>GNN operators that rely on a message passing scheme do not need to be modified since messages are not exchanged between two nodes that belong to different graphs.</p></li><li><p>There is no computational or memory overhead since adjacency matrices are saved in a sparse fashion holding only non-zero entries, <em>i.e.</em>, the edges.</p></li></ol><p>GraphNeuralNetworks.jl can <strong>batch multiple graphs into a single giant graph</strong>:</p></div>

<pre class='language-julia'><code class='language-julia'>vec_gs, _ = first(train_loader)</code></pre>
<pre class="code-output documenter-example-output" id="var-vec_gs">(GraphNeuralNetworks.GNNGraphs.GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}[GNNGraph(19, 40), GNNGraph(23, 54), GNNGraph(11, 22), GNNGraph(16, 34), GNNGraph(10, 20), GNNGraph(20, 44), GNNGraph(23, 54), GNNGraph(25, 56), GNNGraph(16, 34), GNNGraph(17, 38)  …  GNNGraph(17, 38), GNNGraph(20, 46), GNNGraph(22, 50), GNNGraph(17, 38), GNNGraph(20, 44), GNNGraph(26, 60), GNNGraph(20, 46), GNNGraph(13, 26), GNNGraph(12, 24), GNNGraph(10, 20)], Bool[0 0 … 0 1; 1 1 … 1 0])</pre>

<pre class='language-julia'><code class='language-julia'>MLUtils.batch(vec_gs)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash102363">GNNGraph:
    num_nodes = 1168
    num_edges = 2578
    num_graphs = 64
    ndata:
        x =&gt; 7×1168 Matrix{Float32}</pre>


<div class="markdown"><p>Each batched graph object is equipped with a <strong><code>graph_indicator</code> vector</strong>, which maps each node to its respective graph in the batch:</p><p class="tex">$$\textrm{graph-indicator} = [1, \ldots, 1, 2, \ldots, 2, 3, \ldots ]$$</p></div>


```
## Training a Graph Neural Network (GNN)
```@raw html
<div class="markdown">
<p>Training a GNN for graph classification usually follows a simple recipe:</p><ol><li><p>Embed each node by performing multiple rounds of message passing</p></li><li><p>Aggregate node embeddings into a unified graph embedding (<strong>readout layer</strong>)</p></li><li><p>Train a final classifier on the graph embedding</p></li></ol><p>There exists multiple <strong>readout layers</strong> in literature, but the most common one is to simply take the average of node embeddings:</p><p class="tex">$$\mathbf{x}_{\mathcal{G}} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathcal{x}^{(L)}_v$$</p><p>GraphNeuralNetworks.jl provides this functionality via <code>GlobalPool(mean)</code>, which takes in the node embeddings of all nodes in the mini-batch and the assignment vector <code>graph_indicator</code> to compute a graph embedding of size <code>[hidden_channels, batchsize]</code>.</p><p>The final architecture for applying GNNs to the task of graph classification then looks as follows and allows for complete end-to-end training:</p></div>

<pre class='language-julia'><code class='language-julia'>function create_model(nin, nh, nout)
    GNNChain(GCNConv(nin =&gt; nh, relu),
             GCNConv(nh =&gt; nh, relu),
             GCNConv(nh =&gt; nh),
             GlobalPool(mean),
             Dropout(0.5),
             Dense(nh, nout))
end</code></pre>
<pre class="code-output documenter-example-output" id="var-create_model">create_model (generic function with 1 method)</pre>


<div class="markdown"><p>Here, we again make use of the <code>GCNConv</code> with <span class="tex">$\mathrm{ReLU}(x) = \max(x, 0)$</span> activation for obtaining localized node embeddings, before we apply our final classifier on top of a graph readout layer.</p><p>Let's train our network for a few epochs to see how well it performs on the training as well as test set:</p></div>

<pre class='language-julia'><code class='language-julia'>function eval_loss_accuracy(model, data_loader, device)
    loss = 0.
    acc = 0.
    ntot = 0
    for (g, y) in data_loader
        g, y = MLUtils.batch(g) |&gt; device, y |&gt; device
        n = length(y)
        ŷ = model(g, g.ndata.x)
        loss += logitcrossentropy(ŷ, y) * n 
        acc += mean((ŷ .&gt; 0) .== y) * n
        ntot += n
    end 
    return (loss = round(loss/ntot, digits=4), acc = round(acc*100/ntot, digits=2))
end</code></pre>
<pre class="code-output documenter-example-output" id="var-eval_loss_accuracy">eval_loss_accuracy (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function train!(model; epochs=200, η=1e-2, infotime=10)
    # device = Flux.gpu # uncomment this for GPU training
    device = Flux.cpu
    model = model |&gt; device
    ps = Flux.params(model)
    opt = Adam(1e-3)
    

    function report(epoch)
        train = eval_loss_accuracy(model, train_loader, device)
        test = eval_loss_accuracy(model, test_loader, device)
        @info (; epoch, train, test)
    end
    
    report(0)
    for epoch in 1:epochs
        for (g, y) in train_loader
            g, y = MLUtils.batch(g) |&gt; device, y |&gt; device
            gs = Flux.gradient(ps) do
                ŷ = model(g, g.ndata.x)
                logitcrossentropy(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        epoch % infotime == 0 && report(epoch)
    end
end</code></pre>
<pre class="code-output documenter-example-output" id="var-train!">train! (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>begin
    nin = 7  
    nh = 64
    nout = 2
    model = create_model(nin, nh, nout)
    train!(model)
end</code></pre>



<div class="markdown"><p>As one can see, our model reaches around <strong>74% test accuracy</strong>. Reasons for the fluctuations in accuracy can be explained by the rather small dataset (only 38 test graphs), and usually disappear once one applies GNNs to larger datasets.</p><h2>(Optional) Exercise</h2><p>Can we do better than this? As multiple papers pointed out (<a href="https://arxiv.org/abs/1810.00826">Xu et al. (2018)</a>, <a href="https://arxiv.org/abs/1810.02244">Morris et al. (2018)</a>), applying <strong>neighborhood normalization decreases the expressivity of GNNs in distinguishing certain graph structures</strong>. An alternative formulation (<a href="https://arxiv.org/abs/1810.02244">Morris et al. (2018)</a>) omits neighborhood normalization completely and adds a simple skip-connection to the GNN layer in order to preserve central node information:</p><p class="tex">$$\mathbf{x}_i^{(\ell+1)} = \mathbf{W}^{(\ell + 1)}_1 \mathbf{x}_i^{(\ell)} + \mathbf{W}^{(\ell + 1)}_2 \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j^{(\ell)}$$</p><p>This layer is implemented under the name <code>GraphConv</code> in GraphNeuralNetworks.jl.</p><p>As an exercise, you are invited to complete the following code to the extent that it makes use of <code>GraphConv</code> rather than <code>GCNConv</code>. This should bring you close to <strong>82% test accuracy</strong>.</p></div>


```
## Conclusion
```@raw html
<div class="markdown">
<p>In this chapter, you have learned how to apply GNNs to the task of graph classification. You have learned how graphs can be batched together for better GPU utilization, and how to apply readout layers for obtaining graph embeddings rather than node embeddings.</p></div>

<!-- PlutoStaticHTML.End -->
```

