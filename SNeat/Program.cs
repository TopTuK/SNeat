using SNeat.Model;
using SNeat.Model.cnn;
using System;
using System.Collections.Generic;
using System.IO;

namespace SNeat
{
    class Program
    {
        static void Main(string[] args)
        {
            //XorTest();
            //MnistTest();
            CnnMnistTest();
            //ConvLayerTest();
            Console.WriteLine("Meow!");
        }

        static void ConvLayerTest()
        {
            double[,] testData = new double[6, 6]
            {
                { 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 1, 0, 0 },
                { 0, 0, 0, 0, 0, 1 }
            };

            double[,] kernel = new double[3, 3]
            {
                { 0, 0, 0 },
                { 0, 0, 0 },
                { 0, 0, 2 }
            };

            CnnConvLayer convLayer = new CnnConvLayer(new List<double[,]>() { kernel }, 3);
            convLayer.Evaluate(new List<double[,]>() { testData });
            var output = convLayer.Data[0];

            for(int i = 0; i < output.GetLength(0); i++)
            {
                for(int j = 0; j < output.GetLength(1); j++)
                {
                    Console.Write(output[i, j]);
                }
                Console.WriteLine();
            }

            CnnMaxPoolLayer cnnMaxPoolLayer = new CnnMaxPoolLayer(2);
            cnnMaxPoolLayer.Evaluate(convLayer.Data);
            var mpOutput = cnnMaxPoolLayer.Data[0];
            for (int i = 0; i < mpOutput.GetLength(0); i++)
            {
                for (int j = 0; j < mpOutput.GetLength(1); j++)
                {
                    Console.Write(mpOutput[i, j]);
                }
                Console.WriteLine();
            }

            CnnSoftMaxLayer softMaxLayer = new CnnSoftMaxLayer(4, 2);
            softMaxLayer.Evaluate(cnnMaxPoolLayer.Data);
            var smOutput = softMaxLayer.Data[0];
            for (int i = 0; i < smOutput.GetLength(0); i++)
            {
                for (int j = 0; j < smOutput.GetLength(1); j++)
                {
                    Console.Write(smOutput[i, j]);
                }
                Console.WriteLine();
            }
        }

        static void CnnMnistTest()
        {
            Console.WriteLine("Convolution Mnist challenge accepted!");
            Console.WriteLine();

            var trainSet = new TrainingSet();

            CnnMnistModel cnnModel = new CnnMnistModel(trainSet);
            cnnModel.Search(5, 100);
        }

        static void MnistTest()
        {
            Console.WriteLine("Mnist challenge accepted!");
            Console.WriteLine();

            var trainSet = new TrainingSet();

            PopulationParameters populationParameters = new PopulationParameters(5, 100, 1, 2);
            MutationParameters mutationParameters = new MutationParameters();
            NetworkParameters networkParameters = new NetworkParameters(MnistDataset.ImageSize, MnistDataset.ClassCount);
            MnistModel mnistModel = new MnistModel(populationParameters, mutationParameters, networkParameters, trainSet);

            mnistModel.Search(2, 0.1, 100);

            Console.WriteLine();
            Console.WriteLine("Best individual");
            Console.WriteLine(mnistModel.BestIndividual);

            Console.WriteLine();
        }

        static void XorTest()
        {
            Console.WriteLine("XOR challenge accepted!");
            Console.WriteLine();

            NetworkParameters networkParameters = new NetworkParameters(2, 1);
            PopulationParameters populationParameters = new PopulationParameters(20, 10000, 5, 10);
            MutationParameters mutationParameters = new MutationParameters();

            double[][] XorTruthTable =
            {
                new double[] {0.0f, 0.0f, 0.0f},
                new double[] {0.0f, 1.0f, 1.0f},
                new double[] {1.0f, 0.0f, 1.0f},
                new double[] {1.0f, 1.0f, 0.0f}
            };

            XorModel xorModel = new XorModel(populationParameters, mutationParameters, XorTruthTable);
            var iteration = xorModel.Search(100, 0.00001f);

            Console.WriteLine();
            Console.WriteLine($"BEST. Iteration = {iteration} Error = {xorModel.BestIndividual.Error}");
            var net = xorModel.BestIndividual.Genome.Network;
            foreach (var xorData in XorTruthTable)
            {
                net.Inputs[0] = xorData[0];
                net.Inputs[1] = xorData[1];

                net.Activate();

                Console.WriteLine($"[{xorData[0]} {xorData[1]}] = {net.Outputs[0]}");
            }
            Console.WriteLine();

            xorModel.BestIndividual.Genome.PrintGenomeStructure();

            Console.WriteLine("Dump to File");
            File.WriteAllText("xor_net_genome.json", xorModel.BestIndividual.Genome.DumpToJson());

            Console.WriteLine();
        }
    }
}
