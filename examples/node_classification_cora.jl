# An example of semi-supervised node classification

module Trans

using Flux
using Flux: onecold, onehotbatch
using Flux.Losses: logitcrossentropy
using GraphNeuralNetworks
using MLDatasets: Cora
using Statistics, Random
using CUDA
CUDA.allowscalar(false)

function eval_loss_accuracy(X, y, mask, model, g)
    ŷ = model(g, X)
    l = logitcrossentropy(ŷ[:,mask], y[:,mask])
    acc = mean(onecold(ŷ[:,mask]) .== onecold(y[:,mask]))
    return (loss = round(l, digits=4), acc = round(acc*100, digits=2))
end

# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1f-3             # learning rate
    epochs = 150          # number of epochs
    seed = 17             # set seed > 0 for reproducibility
    usecuda = true      # if true use cuda (if available)
    nhidden = 128        # dimension of hidden features
    infotime = 10 	     # report every `infotime` epochs
end

function train(; kws...)
    args = Args(; kws...)

    args.seed > 0 && Random.seed!(args.seed)
    
    if args.usecuda && CUDA.functional()
        device = gpu
        args.seed > 0 && CUDA.seed!(args.seed)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # LOAD DATA
    dataset = Cora()
    classes = dataset.metadata["classes"]
    g = mldataset2gnngraph(dataset) |> device
    X = g.ndata.features
    y = onehotbatch(g.ndata.targets |> cpu, classes) |> device # remove when https://github.com/FluxML/Flux.jl/pull/1959 tagged
    (; train_mask, val_mask, test_mask) = g.ndata
    ytrain = y[:,train_mask]

    nin, nhidden, nout = size(X,1), args.nhidden, length(classes)
    
    ## DEFINE MODEL
    model1 = GNNChain(GCNConv(nin => nhidden, relu),
                     GCNConv(nhidden => nhidden, relu), 
                     Dense(nhidden, nout))  |> device
    model2 = GNNChain(MHAConv(nin => nhidden, relu),
                     MHAConv(nhidden => nhidden, relu), 
                     Dense(nhidden, nout))  |> device
    model = model2

    ps = Flux.params(model)
    opt = Adam(args.η)

    display(g)
    
    ## LOGGING FUNCTION
    function report(epoch)
        train = eval_loss_accuracy(X, y, train_mask, model, g)
        test = eval_loss_accuracy(X, y, test_mask, model, g)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    end
    
    ## TRAINING
    report(0)
    for epoch in 1:args.epochs
        gs = Flux.gradient(ps) do
            ŷ = model(g, X)
            logitcrossentropy(ŷ[:,train_mask], ytrain)
        end

        Flux.Optimise.update!(opt, ps, gs)
        
        epoch % args.infotime == 0 && report(epoch)
    end
end


struct TransformerGNN
    dense1
    conv1
    conv2
    conv3
    dense2
end

function TransformerGNN(in, dh, out, ein, heads)
    TransformerGNN(Dense(in => dh), 
        TransformerConv((dh, ein) => dh; heads), 
        TransformerConv((dh, ein) => dh; heads), 
        TransformerConv((dh, ein) => dh; heads), 
        Dense(dh*heads => out))
end

function (model::TransformerGNN)(g::GNNGraph, x, e)
    x = model.dense1(x)
    x = model.TransformerConv(g, x, e)
    x = model.TransformerConv(g, x, e)
    x = model.TransformerConv(g, x, e)
    x = model.dense2(x)
end

Flux.@functor TransformerGNN

function trans()
    adj =  [ 0 1 1 1
             1 0 1 0
             0 1 0 1
             1 0 1 0]
    adj1 = [0;;]
    adj2 = [0 0 0 1
           0 0 0 0
           0 0 0 1
           1 0 1 0]
    N = size(adj, 1)
    Ne = sum(adj)
    
    ein = 3
    out = 2
    dh = 4
    heads = 3
    in = out * heads
    @show N, Ne, in, ein, out
    

    ge = GNNGraph(adj, ndata=rand(Float32, in, N), 
        edata=rand(Float32, ein, Ne), 
        graph_type=:sparse)
    xe = node_features(ge)
    ee = edge_features(ge)

    layer = TransformerConv((in, 0) => out; heads, concat=true, 
        add_self_loops=false, root_weight=false, ff_channels=30, skip_connection=true, batch_norm=true)
    println(layer)
    y = layer(ge, xe) #, ee)
    @info "TransformerConv" y

    model2 = MHA2GNN(in, dh, out, ein, heads)
    y = model2(ge, xe, ee)
    @info "TransformerGNN" y

end

end  # module

Trans.trans();

# train()

