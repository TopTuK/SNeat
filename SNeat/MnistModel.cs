using SNeat.Model;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SNeat
{
    public class MnistIndividual : IComparable<MnistIndividual>
    {
        public INeatGenome Genome { get; }
        public double Error { get; private set; } // MeanCrossEntropy
        public float Fitness { get; private set; } // Correct recognized digits

        public MnistIndividual(INeatGenome genome)
        {
            Genome = genome;
            Error = double.NaN;
            Fitness = float.NaN;
        }

        public void EvaluateError(IReadOnlyCollection<(double[] image, byte label)> mnistData)
        {
            bool IsMax(IReadOnlyList<double> output, int idx)
            {
                for (var i = 0; i < output.Count; ++i)
                {
                    if (output[i] > output[idx])
                        return false;
                }

                return true;
            }

            Error = double.NaN;
            Fitness = float.NaN;

            var network = Genome.Network;

            double errSum = 0.0f;
            var numCorrect = 0.0f;
            foreach (var (image, label) in mnistData)
            {
                for (int i = 0; i < MnistDataset.ImageSize; i++)
                {
                    network.Inputs[i] = image[i];
                }

                network.Activate();

                errSum += Math.Log(network.Outputs[label]);
                if (IsMax(network.Outputs, label)) numCorrect++;
            }

            Error = -1f * errSum / mnistData.Count;
            Fitness = numCorrect / mnistData.Count;
        }

        public int CompareTo(MnistIndividual other)
        {
            if (ReferenceEquals(this, other)) return 0;
            if (ReferenceEquals(null, other)) return 1;

            var meanCrossEntropyErrorComparison = other.Error.CompareTo(Error);
            if (meanCrossEntropyErrorComparison != 0) return meanCrossEntropyErrorComparison;

            return Fitness.CompareTo(other.Fitness);
        }

        public override string ToString()
        {
            return $"F: {Fitness:0.0000}, E: {Error:00.0000}, NC: {Genome.NodeList.Count}, CC: {Genome.Connections.Count}";
        }
    }

    public class MnistModel
    {
        private const double TrainSpeed = 0.7f;
        private const double TrainMoment = 0.3f;
        private const double L1Ratio = 0.0f;
        private const double L2Ratio = 0.0f;

        private PopulationParameters _populationParameters;
        private MutationParameters _mutationParameters;
        private NetworkParameters _networkParameters;
        private MnistDataset _dataset;

        private List<MnistIndividual> _population;

        public MnistIndividual BestIndividual { get; private set; }

        public MnistModel(PopulationParameters populationParameters, MutationParameters mutationParameters, 
            NetworkParameters networkParameters, MnistDataset dataset)
        {
            _populationParameters = populationParameters;
            _mutationParameters = mutationParameters;
            _networkParameters = networkParameters;
            _dataset = dataset;

            _population = Enumerable.Range(0, _populationParameters.PopulationSize)
                .AsParallel()
                .Select(i =>
                {
                    var genome = new NeatGenome(networkParameters, _mutationParameters);
                    genome.CreateInitialGenome();

                    return new MnistIndividual(genome);
                })
                .ToList();

            BestIndividual = null;
        }

        private void TrainIndividual(MnistIndividual individual, double[][] trainData)
        {
            for (int i = 0; i < _populationParameters.TrainIterations; i++)
            {
                individual.Genome.Network.Train(trainData, TrainSpeed, TrainMoment);
            }
        }

        private void MutateIndividual(MnistIndividual individual, Random random)
        {
            var chance = random.NextDouble();
            if (chance < _mutationParameters.IndividualMutationChance)
            {
                individual.Genome.MutateStructure();
            }
        }

        private IReadOnlyList<(double[], byte)> GetTrainSubset(Random rand, int trainSubsetCount)
        {
            var trainSubset = _dataset.Data
                .OrderBy(_ => rand.NextDouble())
                .Take(trainSubsetCount)
                .ToList();
            return trainSubset;
        }

        private double[][] GetTrainSubsetData(IReadOnlyList<(double[], byte)> trainSubset)
        {
            var trainSubsetData = trainSubset.
                Select(x =>
                {
                    var (image, label) = x;
                    return image.Concat(MnistDataset.OneHotVectors[label]).ToArray();
                })
                .ToArray();

            return trainSubsetData;
        }

        public int Search(int maxIterations, double treshhold, int trainSubsetCount)
        {
            Random rand = new Random();
            int iteration = 0;

            var trainSubset = GetTrainSubset(rand, trainSubsetCount);
            double[][] trainSubsetData = GetTrainSubsetData(trainSubset);

            Parallel.ForEach(_population, individual => TrainIndividual(individual, trainSubsetData));

            while (iteration < maxIterations)
            {

                Parallel.ForEach(_population, individual => individual.EvaluateError(trainSubset));
                _population.Sort();

                BestIndividual = _population[0];
                Console.WriteLine($"I: {iteration} -> {BestIndividual}");
                if (BestIndividual.Error <= treshhold) break;

                var archiveChilds = _population
                    .Take(_populationParameters.ArchiveSize)
                    .ToList();

                var bestChilds = new List<MnistIndividual>(_populationParameters.BestChildCount);
                bestChilds.AddRange(
                    Enumerable.Range(1, _populationParameters.BestChildCount)
                        .AsParallel()
                        .Select(i => new MnistIndividual(NeatGenome.Crossover(BestIndividual.Genome, _population[i].Genome)))
                        .ToList()
                );

                var randomChildCount = _populationParameters.PopulationSize - (_populationParameters.BestChildCount + _populationParameters.ArchiveSize);
                var randomChilds = Enumerable.Range(0, randomChildCount)
                    .AsParallel()
                    .Select(_ =>
                    {
                        MnistIndividual randomChild = null;
                        var individual1 = _population[rand.Next(_population.Count)];
                        var individual2 = _population[rand.Next(_population.Count)];

                        if (individual1 == individual2)
                        {
                            randomChild = individual1;
                            //randomChild.Genome.MutateStructure();
                        }
                        else
                        {
                            randomChild = new MnistIndividual(NeatGenome.Crossover(individual1.Genome, individual2.Genome));
                        }

                        return randomChild;
                    })
                    .ToList();

                _population = new List<MnistIndividual>(_populationParameters.PopulationSize);
                _population.AddRange(bestChilds);
                _population.AddRange(randomChilds);
                _population.AddRange(archiveChilds);

                // Apply mutations
                Parallel.ForEach(bestChilds, individual => MutateIndividual(individual, rand));
                Parallel.ForEach(randomChilds, individual => MutateIndividual(individual, rand));

                // Train population
                trainSubset = GetTrainSubset(rand, trainSubsetCount);
                trainSubsetData = GetTrainSubsetData(trainSubset);
                Parallel.ForEach(_population, individual => TrainIndividual(individual, trainSubsetData));

                iteration++;
            }

            return iteration;
        }
    }
}
