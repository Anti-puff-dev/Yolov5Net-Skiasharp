using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using SkiaSharp;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Yolov5Net.Scorer.Extensions;
using Yolov5Net.Scorer.Models.Abstract;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Yolov5 scorer.
    /// </summary>
    public class YoloScorer<T> : IDisposable where T : YoloModel
    {
        private readonly T _model;


        private readonly InferenceSession _inferenceSession;


        private float Sigmoid(float value)
        {
            return 1 / (1 + (float)Math.Exp(-value));
        }

        private float[] Xywh2xyxy(float[] source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

        public float Clamp(float value, float min, float max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }


        private SKBitmap ResizeImage(SKImage image)
        {
            var (w, h) = (image.Width, image.Height); // image width and height
            var (xRatio, yRatio) = (_model.Width / (float)w, _model.Height / (float)h); // x, y ratios
            var ratio = Math.Min(xRatio, yRatio); // ratio = resized / original
            var (width, height) = ((int)(w * ratio), (int)(h * ratio)); // roi width and height
            var (x, y) = ((_model.Width / 2) - (width / 2), (_model.Height / 2) - (height / 2)); // roi x and y coordinates

            SKBitmap pre_out = SKBitmap.FromImage(image);
            SKBitmap output = pre_out.Resize(new SKImageInfo(width, height), SKFilterQuality.High);
            SKImage _image = SKImage.FromBitmap(output);


            SKSurface surface = SKSurface.Create(new SKImageInfo(_model.Width, _model.Height));
            surface.Canvas.DrawColor(SKColor.Parse("000000"));
            surface.Canvas.DrawImage(_image, new SKPoint(x, y));

            SKImage img = surface.Snapshot();
            return SKBitmap.FromImage(img);
        }


        private Tensor<float> ExtractPixels(SKImage image)
        {
            var bitmap = SKBitmap.FromImage(image);
            var rectangle = new SKRectI(0, 0, bitmap.Width, bitmap.Height);

            //Console.WriteLine(bitmap.Width + " " + bitmap.Height);

            var tensor = new DenseTensor<float>(new[] { 1, 3, _model.Height, _model.Width });

            Parallel.For(0, bitmap.Height, (y) =>
            {

                Parallel.For(0, bitmap.Width, (x) =>
                {
                    tensor[0, 0, y, x] = bitmap.GetPixel(x, y).Red / 255.0F;
                    tensor[0, 1, y, x] = bitmap.GetPixel(x, y).Green / 255.0F;
                    tensor[0, 2, y, x] = bitmap.GetPixel(x, y).Blue / 255.0F;

                    //Console.WriteLine(bitmap.GetPixel(x, y).Red + " " + bitmap.GetPixel(x, y).Green + " " + bitmap.GetPixel(x, y).Blue);
                });
            });

            return tensor;
        }


        private DenseTensor<float>[] Inference(SKImage image)
        {
            SKBitmap resized = null;

            if (image.Width != _model.Width || image.Height != _model.Height)
            {
                resized = ResizeImage(image); // fit image size to specified input size
            }

            var inputs = new List<NamedOnnxValue> // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", ExtractPixels(SKImage.FromBitmap(resized) ?? image))
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _inferenceSession.Run(inputs); // run inference

            var output = new List<DenseTensor<float>>();

            foreach (var item in _model.Outputs) // add outputs for processing
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
            };

            //Console.WriteLine(output.Count + " " + image.Width + " " + image.Height + " " + _model.Width + " " + _model.Height);

            return output.ToArray();
        }


        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, SKImage image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // left, right pads


            Parallel.For(0, (int)output.Length / _model.Dimensions, (i) =>
            {
                if (output[0, i, 4] <= _model.Confidence) return; // skip low obj_conf results

                Parallel.For(5, _model.Dimensions, (j) =>
                {
                    output[0, i, j] = output[0, i, j] * output[0, i, 4]; // mul_conf = obj_conf * cls_conf
                });


                Parallel.For(5, _model.Dimensions, (k) =>
                {
                    if (output[0, i, k] <= _model.MulConfidence) return; // skip low mul_conf results

                    float xMin = ((output[0, i, 0] - output[0, i, 2] / 2) - xPad) / gain; // unpad bbox tlx to original
                    float yMin = ((output[0, i, 1] - output[0, i, 3] / 2) - yPad) / gain; // unpad bbox tly to original
                    float xMax = ((output[0, i, 0] + output[0, i, 2] / 2) - xPad) / gain; // unpad bbox brx to original
                    float yMax = ((output[0, i, 1] + output[0, i, 3] / 2) - yPad) / gain; // unpad bbox bry to original

                    xMin = Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                    yMin = Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                    xMax = Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                    yMax = Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                    YoloLabel label = _model.Labels[k - 5];

                    var prediction = new YoloPrediction(label, output[0, i, k])
                    {
                        Rectangle = new SKRect(xMin, yMin, xMax, yMax)
                    };

                    result.Add(prediction);
                });
            });

            return result.ToList();
        }

 
        private List<YoloPrediction> ParseSigmoid(DenseTensor<float>[] output, SKImage image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // left, right pads

            Parallel.For(0, output.Length, (i) => // iterate model outputs
            {
                int shapes = _model.Shapes[i]; // shapes per output

                Parallel.For(0, _model.Anchors[0].Length, (a) => // iterate anchors
                {
                    Parallel.For(0, shapes, (y) => // iterate shapes (rows)
                    {
                        Parallel.For(0, shapes, (x) => // iterate shapes (columns)
                        {
                            int offset = (shapes * shapes * a + shapes * y + x) * _model.Dimensions;

                            float[] buffer = output[i].Skip(offset).Take(_model.Dimensions).Select(Sigmoid).ToArray();

                            if (buffer[4] <= _model.Confidence) return; // skip low obj_conf results

                            List<float> scores = buffer.Skip(5).Select(b => b * buffer[4]).ToList(); // mul_conf = obj_conf * cls_conf

                            float mulConfidence = scores.Max(); // max confidence score

                            if (mulConfidence <= _model.MulConfidence) return; // skip low mul_conf results

                            float rawX = (buffer[0] * 2 - 0.5f + x) * _model.Strides[i]; // predicted bbox x (center)
                            float rawY = (buffer[1] * 2 - 0.5f + y) * _model.Strides[i]; // predicted bbox y (center)

                            float rawW = (float)Math.Pow(buffer[2] * 2, 2) * _model.Anchors[i][a][0]; // predicted bbox w
                            float rawH = (float)Math.Pow(buffer[3] * 2, 2) * _model.Anchors[i][a][1]; // predicted bbox h

                            float[] xyxy = Xywh2xyxy(new float[] { rawX, rawY, rawW, rawH });

                            float xMin = Clamp((xyxy[0] - xPad) / gain, 0, w - 0); // unpad, clip tlx
                            float yMin = Clamp((xyxy[1] - yPad) / gain, 0, h - 0); // unpad, clip tly
                            float xMax = Clamp((xyxy[2] - xPad) / gain, 0, w - 1); // unpad, clip brx
                            float yMax = Clamp((xyxy[3] - yPad) / gain, 0, h - 1); // unpad, clip bry

                            YoloLabel label = _model.Labels[scores.IndexOf(mulConfidence)];

                            var prediction = new YoloPrediction(label, mulConfidence)
                            {
                                Rectangle = new SKRect(xMin, yMin, xMax, yMax)
                            };

                            result.Add(prediction);
                        });
                    });
                });
            });

            return result.ToList();
        }

     
        private List<YoloPrediction> ParseOutput(DenseTensor<float>[] output, SKImage image)
        {
            return _model.UseDetect ? ParseDetect(output[0], image) : ParseSigmoid(output, image);
        }


        private List<YoloPrediction> Supress(List<YoloPrediction> items)
        {
            var result = new List<YoloPrediction>(items);

            foreach (var item in items) // iterate every prediction
            {
                foreach (var current in result.ToList()) // make a copy for each iteration
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    SKRect intersection = SKRect.Intersect(rect1, rect2);

                    float intArea = intersection.Width * intersection.Height; // intersection area
                    float unionArea = ( rect1.Width * rect1.Height) + (rect2.Width * rect2.Height) - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    //Console.WriteLine(intArea + " " + unionArea + " " + overlap);

                    if (overlap >= _model.Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result;
        }

   
        public List<YoloPrediction> Predict(SKImage image)
        {
            return Supress(ParseOutput(Inference(image), image));
        }


        public YoloScorer()
        {
            _model = Activator.CreateInstance<T>();
        }


        public YoloScorer(string weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), opts ?? new SessionOptions());
        }


        public YoloScorer(Stream weights, SessionOptions opts = null) : this()
        {
            using (var reader = new BinaryReader(weights))
            {
                _inferenceSession = new InferenceSession(reader.ReadBytes((int)weights.Length), opts ?? new SessionOptions());
            }
        }

        public YoloScorer(byte[] weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(weights, opts ?? new SessionOptions());
        }


        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}
