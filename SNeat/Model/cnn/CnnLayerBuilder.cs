using System;
using System.Collections.Generic;
using System.Text;

namespace SNeat.Model.cnn
{
    public class CnnLayerBuilder : ICnnLayerBuilder
    {
        private List<ICnnLayer> _layers;
        public IReadOnlyList<ICnnLayer> Layers => _layers;

        public CnnLayerBuilder()
        {
            _layers = new List<ICnnLayer>();
        }

        public void AddLayer(ICnnLayer layer)
        {
            _layers.Add(layer);
        }
    }
}
