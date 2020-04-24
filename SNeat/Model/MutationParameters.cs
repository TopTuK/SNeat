using System;
using System.Collections.Generic;
using System.Text;

namespace SNeat.Model
{
    public sealed class MutationParameters
    {
        public double IndividualMutationChance { get; set; } = 0.5f;

        public int MutationCount { get; set; } = 1;

        public int AddNodeChance { get; set; } = 2;
        public int AddConnectionChance { get; set; } = 10;
        public int MutateWeightChance { get; set; } = 0;
        
        public double MutationConnectionWeightChance { get; set; } = 0.3f;
        public double RandomWeightChance { get; set; } = 0.5f;
    }
}
