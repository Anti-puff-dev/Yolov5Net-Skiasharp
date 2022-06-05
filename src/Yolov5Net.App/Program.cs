using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models;

namespace Yolov5Net.App
{
    class Program
    {
        static void Main(string[] args)
        {
            MemoryStream imgStream = new MemoryStream(File.ReadAllBytes("Assets/test3.jpg"));
            SKBitmap bitmap = SKBitmap.Decode(imgStream);

            using var image = SKImage.FromBitmap(bitmap);
            Console.WriteLine("image loaded");

            using var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5n.onnx");
            Console.WriteLine("model loaded");

            List<YoloPrediction> predictions = scorer.Predict(image);
            Console.WriteLine("predictions " + predictions.Count);


            SKSurface surface = SKSurface.Create(new SKImageInfo(image.Width, image.Height));
            surface.Canvas.DrawColor(SKColor.Parse("000000"));
            surface.Canvas.DrawImage(image, new SKPoint(0, 0));


            SKPaint paint = new SKPaint();
            paint.IsStroke = true;
            paint.IsAntialias = true;
            paint.Typeface = SKTypeface.FromFamilyName("Arial");


            foreach (var prediction in predictions) // iterate predictions to draw results
            {
                double score = Math.Round(prediction.Score, 2);
                paint.Color = prediction.Label.Color;
                surface.Canvas.DrawRect(prediction.Rectangle, paint);
                var (x, y) = (prediction.Rectangle.Left - 3, prediction.Rectangle.Top - 23);
                surface.Canvas.DrawText($"{prediction.Label.Name} ({score})", new SKPoint(x, y), paint);
            }

            SKImage img = surface.Snapshot();
            SKBitmap bmp = SKBitmap.FromImage(img);

            using (Stream s = File.Create("Assets/result3.jpg"))
            {
                SKData d = SKImage.FromBitmap(bmp).Encode(SKEncodedImageFormat.Jpeg, 100);
                d.SaveTo(s);
            }
        }
    }
}
