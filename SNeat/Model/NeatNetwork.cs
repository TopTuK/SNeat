using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;

namespace SNeat.Model
{
    internal class NetNode : INetNode
    {
        public NetNode(ActivationType actType)
        {
            ActType = actType;
        }

        public ActivationType ActType { get; }
    }

    internal class NetConnection : INetConnection
    {
        public NetConnection(int targetIdx, double weight)
        {
            TargetIdx = targetIdx;

            Weight = weight;
            DeltaWeight = 0.0f;
        }

        public int TargetIdx { get; }

        public double Weight { get; set; }
        public double DeltaWeight { get; set; }
    }

    public class NeatNetwork : INeatNetwork
    {
        private readonly INetConnection[][] _connections;

        private readonly int _inputCount;
        private readonly int _outputCount;

        private readonly INetNode[] _nodes;

        private readonly double[] _preActivations;
        private readonly double[] _postActivations;

        private readonly double[] _gradient;

        public IList<double> Inputs { get; }
        public IReadOnlyList<double> Outputs { get; }

        public NeatNetwork(int inputCount, int outputCount, INetNode[] nodes, INetConnection[][] connections)
        {
            _inputCount = inputCount;
            _outputCount = outputCount;

            _nodes = nodes;

            _connections = connections;

            _preActivations = new double[connections.Length];
            _postActivations = new double[connections.Length];

            _gradient = new double[connections.Length];

            Enumerable.Range(0, NetworkParameters.BiasNodeCount)
                .Select(i => _postActivations[i] = NetworkParameters.BiasNodeValue)
                .Count();

            Inputs = new ArraySegment<double>(_postActivations, NetworkParameters.BiasNodeCount, inputCount);
            Outputs = new ArraySegment<double>(_postActivations, inputCount + NetworkParameters.BiasNodeCount, outputCount);
        }

        public void Activate()
        {
            Activate(true);
        }

        private void Activate(bool resetPreActivation)
        {
            double CalcActivation(double x, INetNode node)
            {
                switch (node.ActType)
                {
                    case ActivationType.BIAS:
                        return NetworkParameters.BiasNodeValue;
                    case ActivationType.IDENTITY:
                        return x;
                    case ActivationType.LEAKYRELU:
                        return x > 0 ? x : x * 0.001f;
                    case ActivationType.SIGMOID:
                        return (1.0f / (1.0f + Math.Pow(Math.E, (-1.0f * x))));
                    default:
                        throw new ArgumentOutOfRangeException(nameof(node), node, null);
                }
            }

            // Bias & Inputs
            for (int i = 0; i < _inputCount + NetworkParameters.BiasNodeCount; i++)
            {
                for (int j = 0; j < _connections[i].Length; j++)
                {
                    var connection = _connections[i][j];
                    _preActivations[connection.TargetIdx] += _postActivations[i] * connection.Weight;
                }
            }

            // HIDDEN
            for (int i = (NetworkParameters.BiasNodeCount + _inputCount + _outputCount); i < _connections.Length; i++)
            {
                _postActivations[i] = CalcActivation(_preActivations[i], _nodes[i]);

                if(resetPreActivation)
                {
                    _preActivations[i] = 0.0f;
                }

                for (int j = 0; j < _connections[i].Length; j++)
                {
                    var connection = _connections[i][j];
                    _preActivations[connection.TargetIdx] += _postActivations[i] * connection.Weight;
                }
            }

            // OUTPUT
            for (int i = (NetworkParameters.BiasNodeCount + _inputCount); i < (NetworkParameters.BiasNodeCount + _inputCount + _outputCount); i++)
            {
                _postActivations[i] = CalcActivation(_preActivations[i], _nodes[i]);

                if(resetPreActivation)
                {
                    _preActivations[i] = 0.0f;
                }
            }
        }

        public void Train(double[][] samples, double trainSpeed = 0.01f, double trainMoment = 0.01f, double l1Ratio = 0.0f, double l2Ratio = 0.0f)
        {
            foreach(var sample in samples)
            {
                TrainIncremental(sample, trainSpeed, trainMoment, l1Ratio, l2Ratio);
            }
        }

        private void TrainIncremental(double[] samples, double trainSpeed, double trainMoment, double l1Ratio, double l2Ratio)
        {
            for (int i = 0; i < _inputCount; i++) Inputs[i] = samples[i];

            Activate(false);

            PropagateError(new ArraySegment<double>(samples, _inputCount, _outputCount),
                trainSpeed, trainMoment,
                l1Ratio, l2Ratio);
        }

        private void PropagateError(IReadOnlyList<double> truthData, double trainSpeed, double trainMoment, double l1Ratio, double l2Ratio)
        {
            // Output
            PropagateOutputError(truthData);

            // Hidden: from Output to Input
            for(int i = _connections.Length - 1; i >= (NetworkParameters.BiasNodeCount + _inputCount + _outputCount); i--)
            {
                PropagateInnerError(i, trainSpeed, trainMoment, l1Ratio, l2Ratio);
            }

            // Input & Bias
            for(int i = NetworkParameters.BiasNodeCount + _inputCount - 1; i >= 0; i--)
            {
                PropageteInputError(i, trainSpeed, trainMoment, l1Ratio, l2Ratio);
            }
        }

        private void PropagateOutputError(IReadOnlyList<double> truthData)
        {
            double CalcDerivative(double pre, double post, INetNode node)
            {
                switch (node.ActType)
                {
                    case ActivationType.IDENTITY:
                        return 1.0f;
                    case ActivationType.SIGMOID:
                        return (1 - post) * post;
                    case ActivationType.LEAKYRELU:
                        return pre < 0f ? 0.001f : 1f;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(node), node, null);
                }
            }

            int idx = 0;
            for (int i = 0; i < _outputCount; i++)
            {
                idx = NetworkParameters.BiasNodeCount + _inputCount + i;

                //_gradient[idx] = (Outputs[i] - truthData[i]);
                _gradient[idx] = (Outputs[i] - truthData[i]) * CalcDerivative(_preActivations[idx], _postActivations[idx], _nodes[idx]);
                //_gradient[idx] = (truthData[i] - Outputs[i]) * CalcDerivative(_preActivations[idx], _postActivations[idx], _nodes[idx]); ;
                _preActivations[idx] = 0.0f;
            }
        }

        private void PropagateInnerError(int index, double trainSpeed, double trainMoment, double l1Ratio, double l2Ratio)
        {
            double CalcDerivative(double pre, double post, INetNode node)
            {
                switch(node.ActType)
                {
                    case ActivationType.IDENTITY:
                        return 1.0f;
                    case ActivationType.SIGMOID:
                        return (1 - post) * post;
                    case ActivationType.LEAKYRELU:
                        return pre < 0f ? 0.001f : 1f;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(node), node, null);
                }
            }

            _gradient[index] = 0.0f;
            for(int i = 0; i < _connections[index].Length; i++)
            {
                var connection = _connections[index][i];

                _gradient[index] += connection.Weight * _gradient[connection.TargetIdx];

                // Get delta weight for each output connection
                var weightDelta = GetWeightDelta(
                    _postActivations[index], // OUT_a <- Current out
                    _gradient[connection.TargetIdx], // DELTA_b <- delta in target node
                    trainSpeed, trainMoment, connection.DeltaWeight,
                    connection.Weight, l1Ratio, l2Ratio);

                //connection.Weight = connection.Weight + weightDelta;
                connection.Weight += (weightDelta + connection.DeltaWeight * trainMoment);
                connection.DeltaWeight = weightDelta;
            }

            // Delta calculation
            _gradient[index] *= CalcDerivative(_preActivations[index], _postActivations[index], _nodes[index]);
            _preActivations[index] = 0.0f;
        }

        private void PropageteInputError(int index, double trainSpeed, double trainMoment, double l1Ratio, double l2Ratio)
        {
            for(int i = 0; i < _connections[index].Length; i++)
            {
                var connection = _connections[index][i];

                var weightDelta = GetWeightDelta(
                    _postActivations[index],
                    _gradient[connection.TargetIdx],
                    trainSpeed, trainMoment, connection.DeltaWeight,
                    connection.Weight, l1Ratio, l2Ratio);

                //connection.Weight = connection.Weight + weightDelta;
                connection.Weight += (weightDelta + connection.DeltaWeight * trainMoment);
                connection.DeltaWeight = weightDelta;
            }
        }

        public IList<double> GetInputGradient()
        {
            double[] inputGradient = new double[_inputCount];

            for(int i = 0; i < _inputCount; i++)
            {
                var idx = NetworkParameters.BiasNodeCount + i;

                inputGradient[i] = 0.0f;
                foreach(var connection in _connections[idx])
                {
                    inputGradient[i] += connection.Weight * _gradient[connection.TargetIdx];
                }
            }

            return inputGradient;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double GetWeightDelta(double input, double gradient, double trainSpeed, double trainMoment, double deltaWeight,
            double weight, double l1Ratio, double l2Ratio)
        {
            return -trainMoment * (input * gradient)
                +l1Ratio * (weight > 0f ? 1f : weight < 0 ? -1f : 0f)
                + l2Ratio * weight; ;

            /*var grad = input * gradient;
            return trainSpeed*grad + trainMoment*deltaWeight
                + l1Ratio * (weight > 0f ? 1f : weight < 0 ? -1f : 0f)
                + l2Ratio * weight;*/
        }
    }
}
