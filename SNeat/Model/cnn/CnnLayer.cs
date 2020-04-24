using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SNeat.Model.cnn
{
    internal class CnnConvLayer : ICnnLayer
    {
        private int _kernelSize;
        private List<double[,]> _kernels;
        private List<double[,]> _layerData;
        private IReadOnlyList<double[,]> _prevLayerData;

        public IReadOnlyList<double[,]> Data => _layerData;

        public CnnConvLayer(int kernelCount, int kernelSize)
        {
            _kernelSize = kernelSize;

            _kernels = new List<double[,]>(kernelCount);
            for(int i = 0; i < kernelCount; i++)
            {
                _kernels.Add(Utils.RandomMatrix(kernelSize));
            }

            _layerData = null;
            _prevLayerData = null;
        }

        public CnnConvLayer(List<double[,]> kernels, int kernelSize)
        {
            _kernelSize = kernelSize;
            _kernels = kernels;
            _layerData = null;
        }

        private IEnumerable<(double[,] region, int x, int y)> IterateRegions(double[,] layerData)
        {
            // 28 - 3 == 25
            for(int i = 0; i <= (layerData.GetLength(0) - _kernelSize); i++)
            {
                // 28 - 3 == 25
                for (int j = 0; j <= (layerData.GetLength(1) - _kernelSize); j++)
                {
                    double[,] region = new double[_kernelSize, _kernelSize];

                    // max == 2
                    for(int x = 0; x < _kernelSize; x++)
                    {
                        for(int y = 0; y < _kernelSize; y++)
                        {
                            region[x, y] = layerData[(i+x), (j+y)];
                        }
                    }

                    yield return (region, i, j);
                }
            }
        }

        private double GetRegionSum(double[,] region, double[,] kernel)
        {
            double sum = 0.0f;

            for(int i = 0; i < _kernelSize; i++)
            {
                for(int j = 0; j < _kernelSize; j++)
                {
                    sum += (region[i, j] * kernel[i, j]);
                }
            }

            return sum;
        }

        public void Evaluate(IReadOnlyList<double[,]> prevLayerData)
        {
            _prevLayerData = prevLayerData;
            _layerData = new List<double[,]>(_kernels.Count * prevLayerData.Count);

            foreach(var prevData in prevLayerData)
            {
                (int LayerWidth, int LayerHeight) layerSize = (prevData.GetLength(0), prevData.GetLength(1));

                foreach(var kernel in _kernels)
                {
                    double[,] layer = new double[layerSize.LayerWidth - 2, layerSize.LayerHeight - 2];

                    foreach(var region in IterateRegions(prevData))
                    {
                        var regionSum = GetRegionSum(region.region, kernel);
                        layer[region.x, region.y] = regionSum;
                    }

                    _layerData.Add(layer);
                }
            }
        }

        public IReadOnlyList<double[,]> PropagateError(IReadOnlyList<double[,]> gradient, double trainSpeed, double trainMoment)
        {
            /*
                Performs a backward pass of the conv layer.
                - d_L_d_out is the loss gradient for this layer's outputs.
                - learn_rate is a float.
            */
            double[,] MultiplyMatrix(double[,] m, double x)
            {
                var matrix = new double[m.GetLength(0), m.GetLength(1)];

                for(int i = 0; i < m.GetLength(0); i++)
                {
                    for(int j = 0; j < m.GetLength(1); j++)
                    {
                        matrix[i, j] = m[i, j] * x;
                    }
                }

                return matrix;
            }

            void AddMatrix(double[,] m1, double[,] m2)
            {
                for(int i = 0; i < m1.GetLength(0); i++)
                {
                    for(int j = 0; j < m1.GetLength(1); j++)
                    {
                        m1[i, j] += m2[i, j];
                    }
                }
            }

            var d_L_d_Filters = new List<double[,]>(_kernels.Count);
            for(int i = 0; i < _kernels.Count; i++)
            {
                d_L_d_Filters.Add(Utils.ZeroMatrix(_kernelSize));
            }

            foreach (var prevData in _prevLayerData)
            {
                foreach (var region in IterateRegions(prevData))
                {
                    int idx = 0;
                    foreach (var dFilter in d_L_d_Filters)
                    {
                        var rGradient = MultiplyMatrix(region.region, gradient[idx][region.x, region.y]);
                        AddMatrix(dFilter, rGradient);

                        idx++;
                    }
                }
            }

            // Update filters
            int kernelIdx = 0;
            foreach(var kernel in _kernels)
            {
                for(int i = 0; i < _kernelSize; i++)
                {
                    for(int j = 0; j < _kernelSize; j++)
                    {
                        kernel[i, j] -= trainSpeed * d_L_d_Filters[kernelIdx][i, j];
                    }
                }

                kernelIdx++;
            }

            /*
                We aren't returning anything here since we use Conv3x3 as
                the first layer in our CNN. Otherwise, we'd need to return
                the loss gradient for this layer's inputs, just like every
                other layer in our CNN.
            */
            return null;
        }
    }

    public class CnnMaxPoolLayer : ICnnLayer
    {
        private int _poolSize;
        private List<double[,]> _layerData;
        private IReadOnlyList<double[,]> _prevLayerData;

        public IReadOnlyList<double[,]> Data => _layerData;

        public CnnMaxPoolLayer(int poolSize)
        {
            _poolSize = poolSize;

            _layerData = null;
            _prevLayerData = null;
        }

        private IEnumerable<(double Max, int MaxX, int MaxY, int X, int Y)> IterateRegions(double[,] layerData, (int LayerWidth, int LayerHeight) layerSize)
        {
            for(int x = 0; x < layerSize.LayerWidth; x++)
            {
                for(int y = 0; y < layerSize.LayerHeight; y++)
                {
                    double max = double.NaN;
                    int maxX = -1;
                    int maxY = -1;

                    for(int i = x * _poolSize; i < (x * _poolSize + _poolSize); i++)
                    {
                        for(int j = y * _poolSize; j < (y * _poolSize + _poolSize); j++)
                        {
                            if(double.IsNaN(max) || layerData[i, j] > max)
                            {
                                max = layerData[i, j];
                                maxX = i;
                                maxY = j;
                            }
                        }
                    }

                    yield return (max, maxX, maxY, x, y);
                }
            }
        }

        public void Evaluate(IReadOnlyList<double[,]> prevLayerData)
        {
            _prevLayerData = prevLayerData;

            _layerData = new List<double[,]>(prevLayerData.Count);

            foreach(var prevData in prevLayerData)
            {
                (int LayerWidth, int LayerHeight) layerSize = (prevData.GetLength(0) / _poolSize, prevData.GetLength(1) / _poolSize);

                double[,] layer = new double[layerSize.LayerWidth, layerSize.LayerHeight];
                foreach(var region in IterateRegions(prevData, layerSize))
                {
                    layer[region.X, region.Y] = region.Max;
                }

                _layerData.Add(layer);
            }
        }

        public IReadOnlyList<double[,]> PropagateError(IReadOnlyList<double[,]> gradient, double trainSpeed, double trainMoment)
        {
            /*
                Performs a backward pass of the maxpool layer.
                Returns the loss gradient for this layer's inputs.
                - d_L_d_out is the loss gradient for this layer's outputs.
            */

            int idx = 0;
            var inputGradient = new List<double[,]>(_prevLayerData.Count);

            foreach (var prevData in _prevLayerData)
            {
                var grad = gradient[idx];

                var layer = new double[prevData.GetLength(0), prevData.GetLength(1)];
                for(int i = 0; i < prevData.GetLength(0); i++)
                {
                    for(int j = 0; j < prevData.GetLength(1); j++)
                    {
                        layer[i, j] = 0.0f;
                    }
                }

                (int LayerWidth, int LayerHeight) layerSize = (prevData.GetLength(0) / _poolSize, prevData.GetLength(1) / _poolSize);

                foreach (var region in IterateRegions(prevData, layerSize))
                {
                    layer[region.MaxX, region.MaxY] = grad[region.X, region.Y];
                }

                inputGradient.Add(layer);
            }

            return inputGradient;
        }
    }

    public class CnnSoftMaxLayer : ICnnLayer
    {
        private List<double[,]> _layerData;

        private int _inputCount;
        private int _outputCount;
        private double[][] _weights;
        private double[] _biases;

        private double[] _lastInput;
        private IReadOnlyList<double[,]> _prevLayerData;

        public IReadOnlyList<double[,]> Data => _layerData;

        // softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10
        public CnnSoftMaxLayer(int inputCount, int outputCount)
        {
            _inputCount = inputCount;
            _outputCount = outputCount;

            _weights = new double[inputCount][];
            for(int i = 0; i < inputCount; i++)
            {
                _weights[i] = new double[outputCount];
                for(int j = 0; j < outputCount; j++)
                {
                    _weights[i][j] = Utils.RandGenerator.NextDouble() / inputCount;
                }
            }

            _biases = new double[outputCount];
            for (int i = 0; i < outputCount; i++) _biases[i] = 0;
        }

        public void Evaluate(IReadOnlyList<double[,]> prevLayerData)
        {
            _prevLayerData = prevLayerData;

            var inputData = new double[_inputCount];
            int idx = 0;
            foreach(var prevData in prevLayerData)
            {
                foreach (var data in prevData)
                {
                    inputData[idx] = data;
                    idx++;
                }
            }
            _lastInput = inputData;

            double[,] layerData = new double[_outputCount, 2];  // Output;- Exp(Out)
            for(int i = 0; i < _outputCount; i++)
            {
                layerData[i, 0] = 0.0f;
                layerData[i, 1] = 0.0f;

                for(int j = 0; j < _inputCount; j++)
                {
                    layerData[i, 0] += inputData[j] * _weights[j][i];
                }
            }

            double expSum = 0.0f;
            for (int i = 0; i < _outputCount; i++)
            {
                layerData[i, 1] = Math.Exp(layerData[i, 0]);
                expSum += layerData[i, 1];
            }

            for (int i = 0; i < _outputCount; i++)
            {
                layerData[i, 1] /= expSum;
            }

            _layerData = new List<double[,]>
            {
                layerData
            };
        }

        public IReadOnlyList<double[,]> PropagateError(IReadOnlyList<double[,]> gradient, double trainSpeed, double trainMoment)
        {
            /*********
            *
            * Performs a backward pass of the softmax layer.
            * Returns the loss gradient for this layer's inputs.
            * - gradient is the loss gradient for this layer's outputs: [*, 0] - network expectation [*, 1] - loss gradiend
            *
            **********/

            (int idx, double loss) GetLabelIndex(double[,] data)
            {
                for(int i = 0; i < data.GetLength(0); i++)
                {
                    if(data[i, 0] > 0.0f)
                    {
                        return (i, data[i, 1]);
                    }
                }

                return (-1, 0.0f);
            }

            // Only 1 gradient[*, 1] of gradient will be nonzero
            // gradient.Count == 1
            double[,] grad = gradient[0];
            var gradLength = _outputCount;//grad.GetLength(0);
            var layerData = _layerData[0];

            var outExp = new double[gradLength];
            double expSum = 0.0f;
            for(int i = 0; i < gradLength; i++)
            {
                outExp[i] = Math.Exp(layerData[i, 0]);
                expSum += outExp[i];
            }

            // Gradients of out[i] against totals
            var label = GetLabelIndex(grad);
            var d_out_d_t = new double[gradLength];
            for(int i = 0; i < gradLength; i++)
            {
                if(i != label.idx)
                {
                    // -t_exp[i] * t_exp / (S ** 2)
                    d_out_d_t[i] = (-1 * outExp[label.idx] * outExp[i]) / (expSum * expSum);
                }
                else
                {
                    // t_exp[i] * (S - t_exp[i]) / (S ** 2)
                    d_out_d_t[i] = outExp[i] * (expSum - outExp[i]) / Math.Pow(expSum, 2);
                }
            }

            // Gradients of loss against totals
            // d_L_d_t = gradient * d_out_d_t
            // В целом, можно и в цикле по d_out_d_t это считать
            var d_L_d_t = new double[gradLength];
            for(int i = 0; i < gradLength; i++)
            {
                d_L_d_t[i] = label.loss * d_out_d_t[i];
            }

            // Gradients of totals against weights/biases/input
            // _lastInput @ d_l_d_t;
            var d_L_d_w = Utils.MultiplyVectors(_lastInput, d_L_d_t); 
            var d_L_d_b = new double[gradLength];

            // d_L_d_b = d_L_d_t * d_t_d_b -> d_t_d_b - 1
            Array.Copy(d_L_d_t, d_L_d_b, gradLength);

            //var d_L_d_inputs = _weights @ d_L_d_t;
            var d_l_d_input = new double[_inputCount];
            for(int i = 0; i < _inputCount; i++)
            {
                d_l_d_input[i] = Utils.AddVectors(_weights[i], d_L_d_t);
            }
            // END Gradients of totals against weights/biases/input

            // Update weight and biases
            //_biases -= trainSpeed * d_L_d_b;
            for (int i = 0; i < _biases.Length; i++)
            {
                _biases[i] -= trainSpeed * d_L_d_b[i];
            }

            //_weights -= trainSpeed * d_L_d_w
            for (int i = 0; i < _inputCount; i++)
            {
                for(int j = 0; j < _outputCount; j++)
                {
                    _weights[i][j] -= trainSpeed * d_L_d_w[i, j];
                }
            }

            // return d_L_d_inputs.reshape(self.last_input_shape)
            return ReshapeInputGradient(d_l_d_input);
        }

        private IReadOnlyList<double[,]> ReshapeInputGradient(double[] input)
        {
            int idx = 0;
            List<double[,]> inputShape = new List<double[,]>(_prevLayerData.Count);

            foreach(var prevLayerData in _prevLayerData)
            {
                var layer = new double[prevLayerData.GetLength(0), prevLayerData.GetLength(1)];

                for(int i = 0; i < prevLayerData.GetLength(0); i++)
                {
                    for(int j = 0; j < prevLayerData.GetLength(1); j++)
                    {
                        layer[i, j] = input[idx];
                        idx++;
                    }
                }

                inputShape.Add(layer);
            }

            return inputShape;
        }
    }

    public class CnnReLuLayer : ICnnLayer
    {
        private IReadOnlyList<double[,]> _prevLayerData;
        private List<double[,]> _layerData;
        public IReadOnlyList<double[,]> Data => _layerData;

        // 26*26*8
        public void Evaluate(IReadOnlyList<double[,]> prevLayerData)
        {
            double ReLu(double x)
            {
                return x > 0.0f ? x : 0.0f;
            }

            _prevLayerData = prevLayerData;

            _layerData = new List<double[,]>(prevLayerData.Count);
            foreach(var layerData in prevLayerData)
            {
                var layer = new double[layerData.GetLength(0), layerData.GetLength(1)];

                for(int i = 0; i < layerData.GetLength(0); i++)
                {
                    for(int j = 0; j < layerData.GetLength(1); j++)
                    {
                        layer[i, j] = ReLu(layerData[i, j]);
                    }
                }

                _layerData.Add(layer);
            }
        }

        // 26*26*8
        public IReadOnlyList<double[,]> PropagateError(IReadOnlyList<double[,]> gradient, double trainSpeed, double trainMoment)
        {
            return gradient;
        }
    }

    public class CnnNeatSoftMaxLayer : ICnnLayer
    {
        private int _inputCount;
        private int _outputCount;
        private INeatNetwork _neatNetwork;

        private IReadOnlyList<double[,]> _prevLayerData;
        private double[] _lastInput;

        private List<double[,]> _layerData;
        public IReadOnlyList<double[,]> Data => _layerData;

        // inputCount: 13*13*8, outputCount: 10
        public CnnNeatSoftMaxLayer(int inputCount, int outputCount)
        {
            _inputCount = inputCount;
            _outputCount = outputCount;

            NetworkParameters netParams = new NetworkParameters(inputCount, outputCount);
            var genome = new NeatGenome(netParams, null);
            genome.CreateInitialGenome();

            _neatNetwork = genome.Network;

            _layerData = null;

            _lastInput = null;
            _prevLayerData = null;
        }

        public void Evaluate(IReadOnlyList<double[,]> prevLayerData)
        {
            _prevLayerData = prevLayerData;

            var inputData = new double[_inputCount];
            int idx = 0;
            foreach (var prevData in prevLayerData)
            {
                foreach (var data in prevData)
                {
                    _neatNetwork.Inputs[idx] = data;
                    idx++;
                }
            }
            _lastInput = inputData;

            _neatNetwork.Activate();

            var layerData = new double[_outputCount, 2];  // Output;- Exp(Out)
            double expSum = 0.0f;
            for (int i = 0; i < _outputCount; i++)
            {
                layerData[i, 0] = _neatNetwork.Outputs[i];
                layerData[i, 1] = Math.Exp(layerData[i, 0]);
                expSum += layerData[i, 1];
            }

            for (int i = 0; i < _outputCount; i++)
            {
                layerData[i, 1] /= expSum;
            }

            _layerData = new List<double[,]>
            {
                layerData
            };
        }

        public IReadOnlyList<double[,]> PropagateError(IReadOnlyList<double[,]> gradient, double trainSpeed, double trainMoment)
        {
            var grad = gradient[0];

            var sample = new double[_inputCount + _outputCount];
            for(int i = 0; i < _inputCount; i++)
            {
                sample[i] = _lastInput[i];
            }
            for(int i = 0; i < _outputCount; i++)
            {
                sample[_inputCount + i] = grad[i, 0];
            }

            _neatNetwork.Train(new List<double[]>() { sample }.ToArray(), trainSpeed, trainMoment);
            var inputGradient = ((NeatNetwork)_neatNetwork).GetInputGradient();

            return ReshapeInputGradient(inputGradient.ToArray());
        }

        private IReadOnlyList<double[,]> ReshapeInputGradient(double[] input)
        {
            int idx = 0;
            List<double[,]> inputShape = new List<double[,]>(_prevLayerData.Count);

            foreach (var prevLayerData in _prevLayerData)
            {
                var layer = new double[prevLayerData.GetLength(0), prevLayerData.GetLength(1)];

                for (int i = 0; i < prevLayerData.GetLength(0); i++)
                {
                    for (int j = 0; j < prevLayerData.GetLength(1); j++)
                    {
                        layer[i, j] = input[idx];
                        idx++;
                    }
                }

                inputShape.Add(layer);
            }

            return inputShape;
        }
    }
}
