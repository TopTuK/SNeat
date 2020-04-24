using System;
using System.Collections.Generic;
using System.Text;

namespace SNeat.Model
{
    public static class GenomeUtils
    {
        /// <summary>
        /// Dump to Console genome structure
        /// </summary>
        /// <param name="genome">Genome</param>
        public static void PrintGenomeStructure(this INeatGenome genome)
        {
            Console.WriteLine("-------------------");
            Console.WriteLine("Genome structure");
            Console.WriteLine();

            Console.WriteLine($"Input node count: {genome.InputNodeCount}");
            Console.WriteLine($"Output node count: {genome.OutputNodeCount}");
            Console.WriteLine($"Node count: {genome.NodeList.Count}");
            Console.WriteLine($"Connections count: {genome.Connections.Count}");
            Console.WriteLine();

            Console.WriteLine("[Connections]");
            foreach(var connection in genome.Connections)
            {
                Console.WriteLine($"Id: {connection.Key} {connection.Value.SourceNodeIdx}->{connection.Value.TargetNodeIdx} {connection.Value.IsActive}");
            }

            Console.WriteLine();
            Console.WriteLine("-------------------");
        }
    }
}
