# An example of semi-supervised node classification

module Mha

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

function mha()
    N = 4
    adj1 =  [0 1 0 1
             1 0 1 0
             0 1 0 1
             1 0 1 0]
    in_channel = 2
    out_channel = 1
    dh = 4
    heads = 2
    g = GNNGraph(adj1, ndata=rand(Float32, in_channel, N), graph_type=:sparse)
    x = node_features(g)

    # layer = MHAConv(in_channel => out_channel; heads)
    # y = layer(g, x)

    model = GNNChain(Dense(in_channel => dh),
        MHAConv(dh => dh; heads),
        Dense(dh => out_channel))
    y = model(g, x)
    @show y

    # y = model(g, x)
    # @show y
end

end  # module

Mha.mha();

# train()

