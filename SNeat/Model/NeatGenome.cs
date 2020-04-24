using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace SNeat.Model
{
    public partial class NeatGenome : INeatGenome
    {
        // TODO: implement IEquatable + Equals + GetHashCode
        private class NodeGene : INodeGene
        {
            public NodeGeneType NodeType { get; }
            public NodeActivationType ActivationType { get; }
            public int Idx { get; }

            internal NodeGene(NodeGeneType nodeType, int idx, NodeActivationType activationType)
            {
                NodeType = nodeType;
                Idx = idx;
                ActivationType = activationType;
            }

            public bool Equals(INodeGene other)
            {
                // Return true if the fields match:
                return (other != null) && (NodeType == other.NodeType) && (ActivationType == other.ActivationType) && (Idx == other.Idx);
            }

            public override bool Equals(object obj)
            {
                // If parameter is null return false.
                if(obj == null)
                {
                    return false;
                }

                // If parameter cannot be cast to INodeGene return false.
                if(obj is INodeGene)
                {
                    this.Equals((INodeGene)obj);
                }
                else
                {
                    return false;
                }

                return base.Equals(obj);
            }

            public override int GetHashCode()
            {
                return (NodeType, ActivationType, Idx).GetHashCode();
            }
        }

        private class ConnectionGene : IConnectionGene
        {
            public int SourceNodeIdx { get; }
            public int TargetNodeIdx { get; }
            public int InnovationNumber { get; }
            public bool IsActive { get; private set; }
            public double Weight { get; private set; }

            public ConnectionGene(int sourceNodeIdx, int targetNodeIdx, int innovationNumber, bool isActive, double weight)
            {
                SourceNodeIdx = sourceNodeIdx;
                TargetNodeIdx = targetNodeIdx;
                InnovationNumber = innovationNumber;
                IsActive = isActive;
                Weight = weight;
            }

            public ConnectionGene(IConnectionGene connectionGene)
            {
                SourceNodeIdx = connectionGene.SourceNodeIdx;
                TargetNodeIdx = connectionGene.TargetNodeIdx;
                Weight = connectionGene.Weight;
                InnovationNumber = connectionGene.InnovationNumber;
                IsActive = connectionGene.IsActive;
            }

            public void DisableConnection()
            {
                IsActive = false;
            }

            public void RandomWeight(Random random)
            {
                Weight = (random.NextDouble() * 2 - 1);
            }

            public void PeturbWeight(Random random)
            {
                Weight += (random.NextDouble() - 0.5f) * 0.5f;
            }
        }

        private Random _randomGenerator;

        private InnovationTracker _innovationTracker;
        private MutationParameters _mutationParameters;

        private readonly List<INodeGene> _nodeList;

        private readonly List<int> _connectionList;
        private readonly Dictionary<int, IConnectionGene> _connections;

        private readonly List<(int, int)> _vacantConnections;

        private INeatNetwork _neatNetwork;
        private bool _needBuildNetwork;

        public MutationParameters MutationParameters => _mutationParameters;
        public NetworkParameters NetworkParameters => new NetworkParameters(InputNodeCount, OutputNodeCount);

        public int InputNodeCount { get; }
        public int OutputNodeCount { get; }

        public IReadOnlyList<INodeGene> NodeList => _nodeList;
        public IReadOnlyList<int> ConnectionList => _connectionList;
        public IReadOnlyDictionary<int, IConnectionGene> Connections => _connections;
        public IReadOnlyList<(int, int)> VacantConnections => _vacantConnections;

        public INeatNetwork Network
        {
            get
            {
                if (_needBuildNetwork)
                {
                    _neatNetwork = BuildNetwork();
                    _needBuildNetwork = false;
                }

                return _neatNetwork;
            }
        }

        private NeatGenome()
        {
            _randomGenerator = new Random();

            _neatNetwork = null;
            _needBuildNetwork = true;
        }

        public NeatGenome(NetworkParameters networkParameters, MutationParameters mutationParameters)
            : this()
        {
            _innovationTracker = InnovationTracker.GetTracker(networkParameters);
            _mutationParameters = mutationParameters;
            
            InputNodeCount = networkParameters.InputNodeCount;
            OutputNodeCount = networkParameters.OutputNodeCount;

            _nodeList = new List<INodeGene>(NetworkParameters.BiasNodeCount + InputNodeCount + OutputNodeCount);

            // Количество связей по умолчанию - каждый вход с каждым выходом.
            _connectionList = new List<int>(InputNodeCount * OutputNodeCount);
            _connections = new Dictionary<int, IConnectionGene>(InputNodeCount * OutputNodeCount);

            // Количество вакантных связей - это количество связей Bias -> Output
            _vacantConnections = new List<(int, int)>(OutputNodeCount);
        }

        public NeatGenome(INeatGenome genome)
            : this()
        {
            _innovationTracker = InnovationTracker.GetTracker(genome.NetworkParameters);
            _mutationParameters = genome.MutationParameters;

            InputNodeCount = genome.InputNodeCount;
            OutputNodeCount = genome.OutputNodeCount;

            _nodeList = new List<INodeGene>(NetworkParameters.BiasNodeCount + InputNodeCount + OutputNodeCount);
            foreach(var genNode in genome.NodeList)
            {
                _nodeList.Add(new NodeGene(genNode.NodeType, genNode.Idx, genNode.ActivationType));
            }

            _connectionList = new List<int>(InputNodeCount * OutputNodeCount);
            foreach(var connectionIdx in genome.ConnectionList)
            {
                _connectionList.Add(connectionIdx);
            }

            _connections = new Dictionary<int, IConnectionGene>(InputNodeCount * OutputNodeCount);
            foreach(var connection in genome.Connections)
            {
                _connections.Add(connection.Key, new ConnectionGene(connection.Value));
            }

            _vacantConnections = new List<(int, int)>(OutputNodeCount);
            foreach(var vacantConnection in genome.VacantConnections)
            {
                _vacantConnections.Add((vacantConnection.Item1, vacantConnection.Item2));
            }
        }

        private double GenerateWeight()
        {
            return ((_randomGenerator.NextDouble() * 2) - 1);
        }

        public void CreateInitialGenome()
        {
            var biasNodes = Enumerable.Range(0, NetworkParameters.BiasNodeCount)
                .Select(i => new NodeGene(NodeGeneType.BIAS, i, NodeActivationType.NONE))
                .ToList();

            var inputNodes = Enumerable.Range(NetworkParameters.BiasNodeCount, InputNodeCount)
                .Select(i => new NodeGene(NodeGeneType.INPUT, i, NodeActivationType.NONE))
                .ToList();

            var outputNodes = Enumerable.Range(InputNodeCount + NetworkParameters.BiasNodeCount, OutputNodeCount)
                .Select(i => new NodeGene(NodeGeneType.OUTPUT, i, NodeActivationType.SIGMOID))
                .ToList();

            _nodeList.AddRange(biasNodes);
            _nodeList.AddRange(inputNodes);
            _nodeList.AddRange(outputNodes);

            // i - OutputIdx
            for(int i = InputNodeCount + NetworkParameters.BiasNodeCount; i <= (InputNodeCount+OutputNodeCount); i++)
            {
                // j - InputIdx
                for(int j = NetworkParameters.BiasNodeCount; j <= InputNodeCount; j++)
                {
                    double weight = GenerateWeight();
                    int innovation = _innovationTracker.AddInnovationConnection(j, i);

                    _connectionList.Add(innovation);
                    _connections.Add(innovation, new ConnectionGene(j, i, innovation, true, weight));
                }
            }

            // Setup vacant connections
            // each Bias -> each Output
            for(int i = 0; i < NetworkParameters.BiasNodeCount; i++)
            {
                for(int j = InputNodeCount + NetworkParameters.BiasNodeCount; j <= (InputNodeCount + OutputNodeCount); j++)
                {
                    _vacantConnections.Add((i, j));
                }
            }
        }

        public void MutateStructure()
        {
            _needBuildNetwork = true;

            for (int i = 0; i < _mutationParameters.MutationCount; i++)
            {
                // Добавление связей возможно только при условии наличия вакантных связей
                if (_vacantConnections.Count > 0)
                {
                    int chance = _randomGenerator.Next(_mutationParameters.AddConnectionChance 
                        + _mutationParameters.AddNodeChance 
                        + _mutationParameters.MutateWeightChance);

                    if(chance < _mutationParameters.AddConnectionChance)
                    {
                        MutateAddConnection();
                    }
                    else if(chance < _mutationParameters.AddConnectionChance + _mutationParameters.AddNodeChance)
                    {
                        MutateAddNode();
                    }
                    else
                    {
                        MutateConnectionWeight();
                    }
                }
                else // нет вакантных связеей
                {
                    int chance = _randomGenerator.Next(_mutationParameters.AddNodeChance + _mutationParameters.MutateWeightChance);

                    if(chance < _mutationParameters.AddNodeChance)
                    {
                        MutateAddNode();
                    }
                    else
                    {
                        MutateConnectionWeight();
                    }
                }
            }
        }

        private void MutateConnectionWeight()
        {
            foreach(var connectionGene in _connections.Values)
            {
                var p = (_randomGenerator.NextDouble() * 2 - 1);
                if (p <= _mutationParameters.MutationConnectionWeightChance)
                {
                    p = (_randomGenerator.NextDouble() * 2 - 1);
                    if (p < _mutationParameters.RandomWeightChance)
                    {
                        connectionGene.RandomWeight(_randomGenerator);
                    }
                    else
                    {
                        connectionGene.PeturbWeight(_randomGenerator);
                    }
                }
            }
        }

        // Добавление связи к существующим нейронам
        private void MutateAddConnection()
        {
            int vacantConnectionIdx = _randomGenerator.Next(_vacantConnections.Count);
            var vacantConnection = _vacantConnections[vacantConnectionIdx];

            var innovation = _innovationTracker.AddInnovationConnection(vacantConnection.Item1, vacantConnection.Item2);

            _connectionList.Add(innovation);
            bool foundPath = false;//FindConnectionPath(vacantConnection.Item1, vacantConnection.Item2);
            _connections.Add(innovation, new ConnectionGene(vacantConnection.Item1, vacantConnection.Item2, innovation, !foundPath, GenerateWeight()));

            _vacantConnections.Remove(vacantConnection);
        }

        // Добавление нового нейрона посередине имеющейся связи
        private void MutateAddNode()
        {
            var nodeConnections = _connections
                .Where(c => c.Value.SourceNodeIdx >= NetworkParameters.BiasNodeCount)
                .Select(c => c)
                .ToList();

            var connection = nodeConnections[_randomGenerator.Next(nodeConnections.Count)].Value;

            var innovation = _innovationTracker.SplitInnovationConnection(connection.SourceNodeIdx, connection.TargetNodeIdx, _nodeList);

            // Add new hidden node
            var node = new NodeGene(NodeGeneType.HIDDEN, innovation.Item1, NodeActivationType.SIGMOID);

            // Add new vacant connections
            var inputNodes = _nodeList
                .Where(n =>
                {
                    return (n.Idx != connection.SourceNodeIdx) && (n.NodeType != NodeGeneType.OUTPUT);
                })
                .Select(n => n)
                .ToList();
            
            foreach(var n in inputNodes)
            {
                _vacantConnections.Add((n.Idx, node.Idx));
            }

            var outputNodes = _nodeList
                .Where(n => 
                {
                    return (n.Idx != connection.TargetNodeIdx) && (n.NodeType != NodeGeneType.INPUT) && (n.NodeType != NodeGeneType.BIAS);
                })
                .Select(n => n)
                .ToList();

            foreach (var n in outputNodes)
            {
                _vacantConnections.Add((node.Idx, n.Idx));
            }
            // End adding new vacant connection

            // Add new node to node list
            _nodeList.Add(node);

            _connectionList.Add(innovation.Item2);
            _connections.Add(innovation.Item2, new ConnectionGene(connection.SourceNodeIdx, node.Idx, innovation.Item2, true, GenerateWeight()));

            _connectionList.Add(innovation.Item3);
            _connections.Add(innovation.Item3, new ConnectionGene(node.Idx, connection.TargetNodeIdx, innovation.Item3, true, GenerateWeight()));

            connection.DisableConnection();
        }

        private INeatNetwork BuildNetwork()
        {
            NetworkBuilder netBuilder = new NetworkBuilder(this);
            return netBuilder.BuildNetwork();
        }

        public string DumpToJson()
        {
            StringBuilder sb = new StringBuilder();
            StringWriter sw = new StringWriter(sb);

            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                writer.Formatting = Formatting.Indented;

                writer.WriteStartObject();
                writer.WritePropertyName("x-type");
                writer.WriteValue("neat-genome");

                // WRITE NODES
                writer.WritePropertyName("neat-nodes");

                writer.WriteStartArray();
                foreach(var node in _nodeList)
                {
                    writer.WriteStartObject();

                    writer.WritePropertyName("node-idx");
                    writer.WriteValue(node.Idx);

                    writer.WritePropertyName("node-type");
                    writer.WriteValue(node.NodeType);

                    writer.WritePropertyName("node-activation-type");
                    writer.WriteValue(node.ActivationType);

                    writer.WriteEndObject();
                }

                writer.WriteEnd();
                // END WRITE NODES

                // WRITE CONNECTIONS
                writer.WritePropertyName("neat-connections");

                writer.WriteStartArray();
                foreach(var connection in _connections.Values)
                {
                    writer.WriteStartObject();

                    writer.WritePropertyName("conn-active");
                    writer.WriteValue(connection.IsActive);

                    writer.WritePropertyName("conn-innovation");
                    writer.WriteValue(connection.InnovationNumber);

                    writer.WritePropertyName("conn-source-idx");
                    writer.WriteValue(connection.SourceNodeIdx);

                    writer.WritePropertyName("conn-target-idx");
                    writer.WriteValue(connection.TargetNodeIdx);

                    writer.WritePropertyName("conn-weight");
                    writer.WriteValue(connection.Weight);

                    writer.WriteEndObject();
                }
                writer.WriteEnd();
                // END WRITE CONNECTIONS

                writer.WriteEndObject();
            }

            return sb.ToString();
        }
    }
}
