using System;
using System.Collections.Generic;
using System.Text;

namespace SNeat.Model.cnn
{
    public interface ICnnLayer
    {
        IReadOnlyList<double[,]> Data { get; }

        void Evaluate(IReadOnlyList<double[,]> prevLayerData);
        IReadOnlyList<double[,]> PropagateError(IReadOnlyList<double[,]> gradient, double trainSpeed, double trainMoment);
    }

    public interface ICnnLayerBuilder
    {
        IReadOnlyList<ICnnLayer> Layers { get; }
        void AddLayer(ICnnLayer layer);
    }
}
