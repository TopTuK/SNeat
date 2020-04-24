using SNeat.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SNeat
{
    class XorIndividual : IComparable<XorIndividual>
    {
        public INeatGenome Genome { get; }
        public double Error { get; private set; }

        public XorIndividual(INeatGenome genome)
        {
            Genome = genome;
            Error = double.NaN;
        }

        public void EvaluateError(double[][] xorTruthTable)
        {
            var network = Genome.Network;

            Error = double.NaN;
            foreach (var xorData in xorTruthTable)
            {
                network.Inputs[0] = xorData[0];
                network.Inputs[1] = xorData[1];

                network.Activate();

                var netRes = network.Outputs[0];
                var netErr = Math.Pow((xorData[2] - netRes), 2.0f);

                if (double.IsNaN(this.Error) || netErr > this.Error)
                {
                    this.Error = netErr;
                }
            }
        }

        public int CompareTo(XorIndividual other)
        {
            return Error.CompareTo(other.Error);
        }

        public override string ToString()
        {
            return $"Genome = E: {Error} NC: {Genome.NodeList.Count} CC: {Genome.Connections.Count}";
        }
    }

    class XorModel
    {
        private static readonly NetworkParameters NetworkParameters = new NetworkParameters(2, 1);

        /*
        private static readonly double[][] XorTruthTable =
        {
            new double[] {0.0f, 0.0f, 0.0f},
            new double[] {0.0f, 1.0f, 1.0f},
            new double[] {1.0f, 0.0f, 1.0f},
            new double[] {1.0f, 1.0f, 0.0f}
        };
        */

        private readonly double[][] _truthTable;

        private const double TrainSpeed = 0.7f;
        private const double TrainMoment = 0.3f;
        private const double L1Ratio = 0.0f;
        private const double L2Ratio = 0.0f;

        private PopulationParameters _populationParameters;
        private MutationParameters _mutationParameters;

        private List<XorIndividual> _population;

        public XorIndividual BestIndividual { get; private set; }

        public XorModel(PopulationParameters populationParameters, MutationParameters mutationParameters, double[][] truthTable)
        {
            _populationParameters = populationParameters;
            _mutationParameters = mutationParameters;

            _truthTable = truthTable;

            _population = Enumerable.Range(0, _populationParameters.PopulationSize)
                .AsParallel()
                .Select(i =>
                {
                    var genome = new NeatGenome(NetworkParameters, _mutationParameters);
                    genome.CreateInitialGenome();

                    return new XorIndividual(genome);
                })
                .ToList();
        }

        private void TrainIndividual(XorIndividual individual)
        {
            for(int i = 0; i < _populationParameters.TrainIterations; i++)
            {
                individual.Genome.Network.Train(_truthTable, TrainSpeed, TrainMoment);
            }
        }

        private void MutateIndividual(XorIndividual individual, Random random)
        {
            var chance = random.NextDouble();
            if(chance < _mutationParameters.IndividualMutationChance)
            {
                individual.Genome.MutateStructure();
            }
        }

        public int Search(int maxIteration, double treshhold)
        {
            Parallel.ForEach(_population, individual => TrainIndividual(individual));

            Random rand = new Random();
            int iteration = 0;
            while(iteration < maxIteration)
            {
                Parallel.ForEach(_population, individual => individual.EvaluateError(_truthTable));
                _population.Sort();

                BestIndividual = _population[0];
                Console.WriteLine($"I: {iteration} -> {BestIndividual}");
                if (BestIndividual.Error <= treshhold) break;

                var archiveChilds = _population
                    .Take(_populationParameters.ArchiveSize)
                    .ToList();

                var bestChilds = new List<XorIndividual>(_populationParameters.BestChildCount);
                bestChilds.AddRange(
                    Enumerable.Range(1, _populationParameters.BestChildCount)
                        .AsParallel()
                        .Select(i => new XorIndividual(NeatGenome.Crossover(BestIndividual.Genome, _population[i].Genome)))
                        .ToList()
                );

                var randomChildCount = _populationParameters.PopulationSize - (_populationParameters.BestChildCount + _populationParameters.ArchiveSize);
                var randomChilds = Enumerable.Range(0, randomChildCount)
                    .AsParallel()
                    .Select(_ =>
                    {
                        XorIndividual randomChild = null;
                        var individual1 = _population[rand.Next(_population.Count)];
                        var individual2 = _population[rand.Next(_population.Count)];

                        if (individual1 == individual2)
                        {
                            randomChild = individual1;
                            //randomChild.Genome.MutateStructure();
                        }
                        else
                        {
                            randomChild = new XorIndividual(NeatGenome.Crossover(individual1.Genome, individual2.Genome));
                        }

                        return randomChild;
                    })
                    .ToList();

                _population = new List<XorIndividual>(_populationParameters.PopulationSize);
                _population.AddRange(bestChilds);
                _population.AddRange(randomChilds);
                _population.AddRange(archiveChilds);

                // Apply mutations
                Parallel.ForEach(bestChilds, individual => MutateIndividual(individual, rand));
                Parallel.ForEach(randomChilds, individual => MutateIndividual(individual, rand));

                // Train population
                Parallel.ForEach(_population, individual => TrainIndividual(individual));

                iteration++;
            }

            return iteration;
        }
    }
}
