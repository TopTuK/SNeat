using SNeat.Model.cnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SNeat
{
    public class CnnMnistModel
    {
        private MnistDataset _dataset;
        private CnnNetwork _cnnNetwork;

        public CnnMnistModel(MnistDataset dataset)
        {
            _dataset = dataset;

            CnnLayerBuilder layerBuilder = new CnnLayerBuilder();
            layerBuilder.AddLayer(new CnnConvLayer(8, 3)); // 26*26*8
            //layerBuilder.AddLayer(new CnnReLuLayer());
            layerBuilder.AddLayer(new CnnMaxPoolLayer(2)); // 13*13*8
            layerBuilder.AddLayer(new CnnSoftMaxLayer(13 * 13 * 8, MnistDataset.ClassCount)); // 10 - Output
            //layerBuilder.AddLayer(new CnnNeatSoftMaxLayer(13 * 13 * 8, MnistDataset.ClassCount));

            _cnnNetwork = new CnnNetwork(layerBuilder, 0.005f, 0.3f);
            //_cnnNetwork = new CnnNetwork(layerBuilder, 0.1f, 0.3f);
        }

        private double[,] GetImageData(double[] data)
        {
            double[,] image = new double[28, 28];
            int idx = 0;
            for(int i = 0; i < 28; i++)
            {
                for(int j = 0; j < 28; j++)
                {
                    image[i, j] = data[idx];
                    idx++;
                }
            }

            return image;
        }

        private (double loss, int acc) Evaluate(double[,] image, byte label)
        {
            bool IsMax(double[,] data, int idx)
            {
                for (int i = 0; i < data.GetLength(0); i++)
                {
                    if (data[i, 1] > data[idx, 1]) return false;
                }

                return true;
            }

            List<double[,]> input = new List<double[,]>
            {
                image
            };

            _cnnNetwork.Activate(input);

            var output = _cnnNetwork.Output;

            var loss = -1 * Math.Log(output[label, 1]);
            var acc = IsMax(output, label) ? 1 : 0;

            return (loss, acc);
        }

        // https://victorzhou.com/blog/intro-to-cnns-part-2/
        private (double Loss, int Acc) Train((double[], byte) trainData)
        {
            // Evaluate
            var image = GetImageData(trainData.Item1);
            var label = trainData.Item2;
            var result = Evaluate(image, label);
            var netOutput = _cnnNetwork.Output;

            // Train
            var expectations = new double[MnistDataset.ClassCount, 2];
            for (int i = 0; i < MnistDataset.ClassCount; i++)
            {
                expectations[i, 0] = 0.0f;
                expectations[i, 1] = 0.0f;
            }
            expectations[label, 0] = 1.0f;
            expectations[label, 1] = -1 / netOutput[label, 1]; // L = -LN(Pc), Pc - corrent digit

            _cnnNetwork.Train(expectations);

            return (result.loss, result.acc);
        }

        public void Search(int epochCount, int trainSubsetCount)
        {
            double loss = 0.0f;
            int numCorrect = 0;

            int iteration = 0;
            foreach(var trainData in _dataset.Data)
            {
                if(iteration % 100 == 99)
                {
                    Console.WriteLine($"[Step {iteration+1}] Past 100 steps: Average Loss: {loss / 100} | Accuracy: {numCorrect}");
                    loss = 0.0f;
                    numCorrect = 0;
                }

                var result = Train(trainData);
                loss += result.Loss;
                numCorrect += result.Acc;

                iteration++;
            }
        }
    }
}
