using System;
using System.Collections.Generic;
using System.Text;

namespace SNeat.Model
{
    public sealed class NetworkParameters
    {
        public const int BiasNodeCount = 1;
        public const double BiasNodeValue = 1.0f;

        public int InputNodeCount { get; }
        public int OutputNodeCount { get; }

        public NetworkParameters(int inputNodeCount, int outputNodeCount)
        {
            if(inputNodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputNodeCount));
            if(outputNodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(outputNodeCount));

            InputNodeCount = inputNodeCount;
            OutputNodeCount = outputNodeCount;
        }
    }
}
