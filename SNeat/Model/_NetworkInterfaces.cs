using System;
using System.Collections.Generic;
using System.Text;

namespace SNeat.Model
{
    public enum ActivationType
    {
        BIAS = 0,
        IDENTITY,
        SIGMOID,
        LEAKYRELU
    }

    public interface INetNode
    {
        ActivationType ActType { get; }
    }

    public interface INetConnection
    {
        int TargetIdx { get; }

        double Weight { get; set; }
        double DeltaWeight { get; set; }
    }

    public interface INeatNetwork
    {
        IList<double> Inputs { get; }
        IReadOnlyList<double> Outputs { get; }

        void Activate();
        void Train(double[][] samples, double trainSpeed = 0.01f, double trainMoment = 0.01f, double l1Ratio = 0.0f, double l2Ratio = 0.0f);
    }
}
