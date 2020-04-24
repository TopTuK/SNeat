using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SNeat.Model
{
    internal class NetworkBuilder
    {
        private class NodeMap
        {
            public NodeMap(int netNodeId, int neuronId)
            {
                NetNodeId = netNodeId;
                NeuronId = neuronId;
            }

            public int NetNodeId;
            public int NeuronId;
        }

        private INeatGenome _genome;
        private Dictionary<NodeMap, IList<INetConnection>> _netMap;
        private int _nextNetNodeId;

        public NetworkBuilder(INeatGenome genome)
        {
            _genome = genome;
            _nextNetNodeId = NetworkParameters.BiasNodeCount + _genome.InputNodeCount + _genome.OutputNodeCount;
            _netMap = new Dictionary<NodeMap, IList<INetConnection>>(_nextNetNodeId);
        }

        public INeatNetwork BuildNetwork()
        {
            var activeConnections = _genome.Connections
                .Where(c => c.Value.IsActive)
                .Select(c => c.Value)
                .ToList();

            /* // Bias & Input & Output
            for (int i = 0; i < NetworkParameters.BiasNodeCount + _genome.InputNodeCount + _genome.OutputNodeCount; i++)
            {
                var nodeMap = new NodeMap(i, i);
                _netMap.Add(nodeMap, new List<INetConnection>());
            }

            var outputNodeLayer2 = Enumerable.Range(NetworkParameters.BiasNodeCount + _genome.InputNodeCount, _genome.OutputNodeCount)
                .Select(i => _netMap.Keys.First(n => n.NeuronId == i))
                .ToList();
            */

            // Bias & Inputs
            for (int i = 0; i < NetworkParameters.BiasNodeCount + _genome.InputNodeCount; i++)
            {
                _netMap.Add(new NodeMap(i, i), new List<INetConnection>());
            }

            // Outputs
            List<NodeMap> outputNodeLayer = new List<NodeMap>(_genome.OutputNodeCount);
            for (int i = NetworkParameters.BiasNodeCount + _genome.InputNodeCount;
                i < NetworkParameters.BiasNodeCount + _genome.InputNodeCount + _genome.OutputNodeCount; 
                i++)
            {
                var nodeMap = new NodeMap(i, i);
                _netMap.Add(nodeMap, new List<INetConnection>());
                outputNodeLayer.Add(nodeMap);
            }

            var innerConnections = GetInnerConnections(activeConnections, outputNodeLayer);
            if(innerConnections != null && innerConnections.Count > 0)
            {
                var currentNodeLayer = Enumerable.Range(0, NetworkParameters.BiasNodeCount + _genome.InputNodeCount)
                    .Select(i => _netMap.Keys.First(n => n.NeuronId == i))
                    .ToList();

                while(currentNodeLayer.Count > 0)
                {
                    currentNodeLayer = GetNextLayerNodes(currentNodeLayer, innerConnections);
                }

                foreach(var innerConnection in innerConnections)
                {
                    var nodes = _netMap
                    .Where(n => n.Key.NeuronId == innerConnection.SourceNodeIdx)
                    .Select(n => n)
                    .ToList();

                    foreach (var node in nodes)
                    {
                        var targetIdx = _netMap.Keys.First(n => n.NeuronId == innerConnection.TargetNodeIdx).NetNodeId;
                        node.Value.Add(new NetConnection(targetIdx, innerConnection.Weight));
                    }
                }
            }

            var outputConnections = GetOutputConnections(activeConnections, outputNodeLayer);
            foreach(var outputConnection in outputConnections)
            {
                var nodes = _netMap
                    .Where(n => n.Key.NeuronId == outputConnection.SourceNodeIdx)
                    .Select(n => n)
                    .ToList();

                foreach(var node in nodes)
                {
                    node.Value.Add(new NetConnection(outputConnection.TargetNodeIdx, outputConnection.Weight));
                }
            }

            var networkNodes = GetNetworkNodes();
            var networkConnections = GetNetworkConnections();
            return new NeatNetwork(_genome.InputNodeCount, _genome.OutputNodeCount, networkNodes, networkConnections);
        }

        private INetNode[] GetNetworkNodes()
        {
            var netNodes = new INetNode[_netMap.Count];
            int index = 0;
            ActivationType actType = ActivationType.IDENTITY;
            foreach (var nodeMap in _netMap.Keys)
            {
                //var genomeNode = _genome.NodeList[nodeMap.NeuronId];
                var genomeNode = _genome.NodeList.First(n => n.Idx == nodeMap.NeuronId);

                switch (genomeNode.ActivationType)
                {
                    case NodeActivationType.NONE:
                        actType = ActivationType.IDENTITY;
                        break;
                    case NodeActivationType.LEAKYRELU:
                        actType = ActivationType.LEAKYRELU;
                        break;
                    case NodeActivationType.SIGMOID:
                        actType = ActivationType.SIGMOID;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(NodeActivationType), genomeNode.ActivationType, null);
                }
                if (genomeNode.NodeType == NodeGeneType.BIAS) actType = ActivationType.BIAS;

                netNodes[index] = new NetNode(actType);
                index++;
            }

            return netNodes;
        }

        private INetConnection[][] GetNetworkConnections()
        {
            var netNodeConnections = new INetConnection[_netMap.Count][];

            foreach (var netNode in _netMap)
            {
                netNodeConnections[netNode.Key.NetNodeId] = netNode.Value.ToArray();
            }            

            return netNodeConnections;
        }

        private IList<IConnectionGene> GetOutputConnections(IReadOnlyList<IConnectionGene> connections, IReadOnlyList<NodeMap> outputNodeLayer)
        {
            var outConnections = new List<IConnectionGene>(outputNodeLayer.Count);
            foreach(var connection in connections)
            {
                bool found = false;

                foreach(var outputNode in outputNodeLayer)
                {
                    if(connection.TargetNodeIdx == outputNode.NeuronId)
                    {
                        found = true;
                        break;
                    }
                }

                if (found) outConnections.Add(connection);
            }

            return outConnections;

            /*
            var outputConnections = connections
                .Where(c => 
                {
                    return outputNodeLayer.FirstOrDefault(n => n.NeuronId == c.TargetNodeIdx) != null;
                })
                .Select(c => c)
                .ToList();

            return outputConnections;
            */
        }

        private List<IConnectionGene> GetInnerConnections(IReadOnlyList<IConnectionGene> connections, IReadOnlyList<NodeMap> outputNodeLayer)
        {
            var innerConnections = new List<IConnectionGene>(outputNodeLayer.Count);
            foreach (var connection in connections)
            {
                bool found = false;

                foreach (var outputNode in outputNodeLayer)
                {
                    if (connection.TargetNodeIdx == outputNode.NeuronId)
                    {
                        found = true;
                        break;
                    }
                }

                if (!found) innerConnections.Add(connection);
            }

            return innerConnections;

            /*
            var innerConnections = connections
                .Where(c =>
                {
                    return outputNodeLayer.FirstOrDefault(n => n.NeuronId != c.TargetNodeIdx) != null;
                })
                .Select(c => c)
                .ToList();

            return innerConnections;
            */
        }

        private List<NodeMap> GetNextLayerNodes(List<NodeMap> prevLayerNodes, IReadOnlyList<IConnectionGene> innerConnections)
        {
            var connectionNodes = innerConnections
                .Where(c =>
                {
                    return prevLayerNodes.FirstOrDefault(n => n.NeuronId == c.SourceNodeIdx) != null;
                })
                .Select(c => c)
                .GroupBy(c => c.TargetNodeIdx)
                .ToList();

            List<NodeMap> nextNodeLayer = new List<NodeMap>(connectionNodes.Count);
            foreach (var connectionNode in connectionNodes)
            {
                var netMapNode = _netMap.Keys.FirstOrDefault(n => n.NeuronId == connectionNode.Key);

                if(netMapNode == null)
                {
                    var nodeMap = new NodeMap(_nextNetNodeId, connectionNode.Key);

                    _netMap.Add(nodeMap, new List<INetConnection>());
                    nextNodeLayer.Add(nodeMap);

                    _nextNetNodeId++;
                }
            }

            return nextNodeLayer;
        }

        public void PrintNetMap()
        {
            Console.WriteLine("--- NET MAP ---");

            foreach(var netMapNode in _netMap)
            {
                Console.WriteLine($"({netMapNode.Key.NetNodeId} [{netMapNode.Key.NeuronId}])");

                foreach(var connection in netMapNode.Value)
                {
                    Console.WriteLine($"{netMapNode.Key.NetNodeId} -> {connection.TargetIdx} ({connection.Weight})");
                }
            }

            Console.WriteLine("--- END NET MAP ---");
        }
    }
}
