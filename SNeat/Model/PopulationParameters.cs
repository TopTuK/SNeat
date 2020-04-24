using System;
using System.Collections.Generic;
using System.Text;

namespace SNeat.Model
{
    public readonly struct PopulationParameters
    {
        public readonly int PopulationSize;
        public readonly int TrainIterations;

        public readonly int ArchiveSize;
        public readonly int BestChildCount;

        public PopulationParameters(int populationSize, int trainIterations, int archiveSize, int bestChildCount)
        {
            if(populationSize < 1)
                throw new ArgumentException("PopulationSize should be more or equal then 1", nameof(populationSize));

            if (archiveSize < 0)
                throw new ArgumentException("ArchiveSize should be more then 0", nameof(archiveSize));

            if((bestChildCount < 0) || (bestChildCount + archiveSize > populationSize))
                throw new ArgumentException("PopulationSize should be more or equal then 1", nameof(populationSize));

            PopulationSize = populationSize;
            TrainIterations = trainIterations;

            ArchiveSize = archiveSize;
            BestChildCount = bestChildCount;
        }
    }
}
