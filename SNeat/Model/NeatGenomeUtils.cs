using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SNeat.Model
{
    public partial class NeatGenome
    {
        public static INeatGenome Crossover(INeatGenome genome1, INeatGenome genome2)
        {
            var childGenome = new NeatGenome(genome1.NetworkParameters, genome1.MutationParameters);

            var nodes = genome1.NodeList.Union(genome2.NodeList).ToList();
            foreach(var node in nodes)
            {
                childGenome._nodeList.Add(new NodeGene(node.NodeType, node.Idx, node.ActivationType));
            }

            var childConnectionList = genome1.ConnectionList.Union(genome2.ConnectionList).ToList();
            foreach(var connectionIdx in childConnectionList)
            {
                childGenome._connectionList.Add(connectionIdx);
            }

            IConnectionGene connectionGene = null;
            for(int i = 0; i < childConnectionList.Count; i++)
            {
                var childConnectionIdx = childConnectionList[i];

                if (genome1.Connections.ContainsKey(childConnectionIdx))
                {
                    connectionGene = genome1.Connections[childConnectionIdx];
                }
                else if (genome2.Connections.ContainsKey(childConnectionIdx))
                {
                    connectionGene = genome2.Connections[childConnectionIdx];
                }
                else
                    throw new ArgumentOutOfRangeException(nameof(childConnectionList), "Parent genomes don't have child connection");

                childGenome._connections.Add(childConnectionIdx, new ConnectionGene(connectionGene));
            }

            var childVacantConnections = genome1.VacantConnections.Union(genome2.VacantConnections).ToList();
            foreach(var vacantConnection in childVacantConnections)
            {
                if(childGenome.Connections.Values
                    .FirstOrDefault(c => (c.SourceNodeIdx == vacantConnection.Item1) && (c.TargetNodeIdx == vacantConnection.Item2))== null)
                {
                    childGenome._vacantConnections.Add((vacantConnection.Item1, vacantConnection.Item2));
                }
            }

            return childGenome;
        }
    }
}
