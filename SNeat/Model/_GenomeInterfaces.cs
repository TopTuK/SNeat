using System;
using System.Collections.Generic;
using System.Text;

namespace SNeat.Model
{
    public enum NodeGeneType
    {
        BIAS = 0,
        INPUT,
        HIDDEN,
        OUTPUT
    }

    public enum NodeActivationType
    {
        NONE,
        SIGMOID,
        LEAKYRELU
    }

    public interface INodeGene : IEquatable<INodeGene>
    {
        NodeGeneType NodeType { get; }
        NodeActivationType ActivationType { get; }

        int Idx { get; }
    }

    public interface IConnectionGene
    {
        int SourceNodeIdx { get; }
        int TargetNodeIdx { get; }

        int InnovationNumber { get; }
        bool IsActive { get; }

        double Weight { get; }

        void DisableConnection();

        void RandomWeight(Random random);
        void PeturbWeight(Random random);
    }

    public interface INeatGenome
    {
        MutationParameters MutationParameters { get; }
        NetworkParameters NetworkParameters { get; }

        int InputNodeCount { get; }
        int OutputNodeCount { get; }

        IReadOnlyList<INodeGene> NodeList { get; }
        IReadOnlyList<int> ConnectionList { get; }
        IReadOnlyDictionary<int, IConnectionGene> Connections { get; }
        IReadOnlyList<(int, int)> VacantConnections { get; }

        INeatNetwork Network { get; }

        void CreateInitialGenome();
        void MutateStructure();

        string DumpToJson();
    }
}
