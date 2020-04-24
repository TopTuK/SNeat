using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SNeat.Model
{
    internal sealed class InnovationTracker
    {
        private static object _lockObj = new object();
        private static volatile InnovationTracker g_innovationTracker;

        private Dictionary<(int, int), int> _innovationTracker;
        private Dictionary<(int, int), List<int>> _splitConnections;

        private int _nextConnectionNumber;
        private int _nextNeuronNumber;

        private InnovationTracker(NetworkParameters networkParameters)
        {
            var inputCount = networkParameters.InputNodeCount;
            var outputCount = networkParameters.OutputNodeCount;

            _nextConnectionNumber = 0;
            _nextNeuronNumber = inputCount + outputCount;

            _innovationTracker = new Dictionary<(int, int), int>();
            _splitConnections = new Dictionary<(int, int), List<int>>();

            for (var i = NetworkParameters.BiasNodeCount; i < (NetworkParameters.BiasNodeCount + inputCount); i++)
            {
                for (var j = (inputCount + NetworkParameters.BiasNodeCount); j <= (inputCount + outputCount); j++)
                {
                    _nextConnectionNumber++;
                    _innovationTracker.Add((i, j), _nextConnectionNumber);
                }
            }
        }

        public int AddInnovationConnection(int sourceIdx, int targetIdx)
        {
            int innovationNumber = -1;

            var link = (sourceIdx, targetIdx);
            lock (_lockObj)
            {
                if (_innovationTracker.ContainsKey(link))
                {
                    innovationNumber = _innovationTracker[link];
                }
                else
                {
                    _nextConnectionNumber++;
                    _innovationTracker.Add(link, _nextConnectionNumber);
                    innovationNumber = _nextConnectionNumber;
                }
            }

            return innovationNumber;
        }

        public (int, int, int) SplitInnovationConnection(int sourceIdx, int targetIdx, IReadOnlyList<INodeGene> nodeGeneList)
        {
            int neuronIdx = -1, innovation1 = -1, innovation2 = -1;

            var linkIdx = (sourceIdx, targetIdx);

            lock(_lockObj)
            {
                if(_splitConnections.ContainsKey(linkIdx)) // Связь была разделена ранее
                {
                    var connectionNodes = _splitConnections[linkIdx];
                    int newNodeIdx = -1;

                    // Поиск в геноме связующий нейрон
                    foreach(var connectionNode in connectionNodes)
                    {
                        var nodeGene = nodeGeneList.FirstOrDefault(n => n.Idx == connectionNode);
                        if(nodeGene == null)
                        {
                            newNodeIdx = connectionNode;
                            break;
                        }
                    }

                    if(newNodeIdx >= 0) // в геноме нет связующего нейрона
                    {
                        neuronIdx = newNodeIdx;
                        innovation1 = _innovationTracker[(sourceIdx, newNodeIdx)];
                        innovation2 = _innovationTracker[(newNodeIdx, targetIdx)];
                    }
                    else // новый нейрон и новые связи
                    {
                        _nextNeuronNumber++;
                        neuronIdx = _nextNeuronNumber;

                        _nextConnectionNumber++;
                        _innovationTracker.Add((sourceIdx, _nextNeuronNumber), _nextConnectionNumber);
                        innovation1 = _nextConnectionNumber;

                        _nextConnectionNumber++;
                        _innovationTracker.Add((_nextNeuronNumber, targetIdx), _nextConnectionNumber);
                        innovation2 = _nextConnectionNumber;

                        // добавляем в split connections, что есть новый нейрон
                        connectionNodes.Add(_nextNeuronNumber);
                    }
                }
                else // Связь разделяется впервые.
                {
                    _nextNeuronNumber++;
                    neuronIdx = _nextNeuronNumber;

                    _nextConnectionNumber++;
                    _innovationTracker.Add((sourceIdx, _nextNeuronNumber), _nextConnectionNumber);
                    innovation1 = _nextConnectionNumber;

                    _nextConnectionNumber++;
                    _innovationTracker.Add((_nextNeuronNumber, targetIdx), _nextConnectionNumber);
                    innovation2 = _nextConnectionNumber;

                    var connectionNodes = new List<int>(1)
                    {
                        _nextNeuronNumber
                    };
                    _splitConnections.Add(linkIdx, connectionNodes);
                }

                /*
                _nextNeuronNumber++;

                if (_splitConnections.ContainsKey(linkIdx))
                {
                    _splitConnections[linkIdx] = _nextNeuronNumber;
                }
                else
                {
                    _splitConnections.Add(linkIdx, _nextNeuronNumber);
                }

                neuronIdx = _nextNeuronNumber;

                _nextConnectionNumber++;
                _innovationTracker.Add((sourceIdx, _nextNeuronNumber), _nextConnectionNumber);
                innovation1 = _nextConnectionNumber;

                _nextConnectionNumber++;
                _innovationTracker.Add((_nextNeuronNumber, targetIdx), _nextConnectionNumber);
                innovation2 = _nextConnectionNumber;
                */
            }

            return (neuronIdx, innovation1, innovation2);
        }

        public static InnovationTracker GetTracker(NetworkParameters networkParameters)
        {
            if(g_innovationTracker == null)
            {
                lock(_lockObj)
                {
                    if (g_innovationTracker == null) g_innovationTracker = new InnovationTracker(networkParameters);
                }
            }

            return g_innovationTracker;
        }
    }
}
