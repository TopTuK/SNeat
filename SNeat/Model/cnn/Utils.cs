using System;
using System.Collections.Generic;
using System.Text;

namespace SNeat.Model.cnn
{
    static class Utils
    {
        public static Random RandGenerator = new Random();

        public static double[,] RandomMatrix(int matrixSize)
        {
            double[,] matrix = new double[matrixSize, matrixSize];

            double nVar = matrixSize * matrixSize;
            for(int i = 0; i < matrixSize; i++)
            {
                for(int j = 0; j < matrixSize; j++)
                {
                    matrix[i, j] = RandGenerator.NextDouble() / nVar;
                }
            }

            return matrix;
        }

        public static double[,] ZeroMatrix(int matrixSize)
        {
            double[,] matrix = new double[matrixSize, matrixSize];

            for (int i = 0; i < matrixSize; i++)
            {
                for (int j = 0; j < matrixSize; j++)
                {
                    matrix[i, j] = 0.0f;
                }
            }

            return matrix;
        }

        public static double AddVectors(double[] vector1, double[] vector2)
        {
            double sum = 0.0f;

            if (vector1.Length != vector2.Length) throw new ArgumentException("AddVectors");

            for(int i = 0; i < vector1.Length; i++)
            {
                sum += vector1[i] * vector2[i];
            }

            return sum;
        }

        public static double[,] MultiplyVectors(double[] matrix1, double[] matrix2)
        {
            /*
             * [2] * [1, 2] = [2, 4]
             * [3]            [3. 6]
            */
            var result = new double[matrix1.Length, matrix2.Length];
            for(int i = 0; i < matrix1.Length; i++)
            {
                for(int j = 0; j < matrix2.Length; j++)
                {
                    result[i, j] = matrix1[i] * matrix2[j];
                }
            }

            return result;
        }
    }
}
