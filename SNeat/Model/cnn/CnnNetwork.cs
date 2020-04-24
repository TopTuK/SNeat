using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SNeat.Model.cnn
{
    public class CnnNetwork
    {
        private double _trainSpeed;
        private double _trainMoment;
        private ICnnLayerBuilder _layerBuilder;

        public double[,] Output { get; private set; }

        public CnnNetwork(ICnnLayerBuilder layerBuilder, double trainSpeed, double trainMoment)
        {
            _layerBuilder = layerBuilder;

            _trainSpeed = trainSpeed;
            _trainMoment = trainMoment;
        }

        public void Activate(IReadOnlyList<double[,]> input)
        {
            var layers = _layerBuilder.Layers;

            var data = input;
            foreach(var layer in layers)
            {
                layer.Evaluate(data);
                data = layer.Data;
            }
            
            // data[*, 0] - Network Output
            // data[*, 1] - Exp/Sum(Exp)
            Output = data[0];
        }

        public void Train(double[,] expectations)
        {
            IReadOnlyList<double[,]> data = new List<double[,]>() { expectations };

            var layers = _layerBuilder.Layers.Reverse();
            foreach(var layer in layers)
            {
                data = layer.PropagateError(data, _trainSpeed, _trainMoment);
            }
        }
    }
}
