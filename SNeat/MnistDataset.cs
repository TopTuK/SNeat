using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace SNeat
{
    public static partial class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(int));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }

    public abstract class MnistDataset
    {
        public static readonly IReadOnlyList<IReadOnlyList<double>> OneHotVectors = Enumerable.Range(0, ClassCount)
            .Select(i =>
                Enumerable.Range(0, ClassCount)
                    .Select(v => v == i ? (double) 1f : 0f)
                    .ToArray())
            .ToArray();

        public const int ImageSize = 28 * 28;
        public const int ClassCount = 10;

        public IReadOnlyList<(double[], byte)> Data { get; }

        protected MnistDataset(string imagesFileName, string labelsFileName)
        {
            var imagesStream = File.OpenRead(imagesFileName);

            double[] ReadImage()
            {
                Span<byte> imageData = stackalloc byte[ImageSize];
                var read = imagesStream.Read(imageData);
                if (read != ImageSize)
                {
                    throw new InvalidOperationException();
                }

                var image = new double[read];
                for (var i = 0; i < imageData.Length; ++i)
                {
                    image[i] = (double) imageData[i] / byte.MaxValue * 2f - 1f;
                    //image[i] = imageData[i] / byte.MaxValue - 0.5f;
                }

                return image;
            }

            double[][] images = null;
            using (var binaryReader = new BinaryReader(imagesStream, Encoding.Default, false))
            {
                var magic = binaryReader.ReadBigInt32();
                if (magic != 2051)
                {
                    throw new InvalidDataException($"Images file ({imagesFileName}) has invalid format.");
                }

                var imageCount = binaryReader.ReadBigInt32();
                images = new double[imageCount][];

                var rowNumber = binaryReader.ReadBigInt32();
                var colNumber = binaryReader.ReadBigInt32();
                if (rowNumber != colNumber || rowNumber * colNumber != ImageSize)
                {
                    throw new InvalidDataException($"Images file ({imagesFileName}) has invalid format.");
                }

                for (var i = 0; i < imageCount; i++)
                {
                    images[i] = ReadImage();
                }
            }

            using (var binaryReader = new BinaryReader(File.OpenRead(labelsFileName)))
            {
                var magic = binaryReader.ReadBigInt32();
                if (magic != 2049)
                {
                    throw new InvalidDataException($"Labels file ({labelsFileName}) has invalid format.");
                }

                var size = binaryReader.ReadBigInt32();
                if (size != images.Length)
                {
                    throw new InvalidDataException($"Labels file ({labelsFileName}) doesn't match images.");
                }

                var labels = binaryReader.ReadBytes(size);
                if (labels.Any(l => l > 9))
                {
                    throw new InvalidDataException($"Labels file ({labelsFileName}) contains invalid labels.");
                }

                Data = labels.Select((l, i) => (images[i], l)).ToList();
            }
        }
    }

    public class TrainingSet : MnistDataset
    {
        public TrainingSet() : base("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
        {
            Debug.Assert(Data.Count == 60000);
        }
    }
}
