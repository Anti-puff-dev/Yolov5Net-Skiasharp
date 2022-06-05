# Yolov5Net with SkiaSharp
YOLOv5 object detection with ML.NET, ONNX, System.Drawing dependencies removed for running under linux with SkiaSharp library, AWS Serverless don't run System.Drawing

![example](https://github.com/mentalstack/yolov5-net/blob/master/img/result.jpg?raw=true)

## Installation

Run this line from Package Manager Console:

```
Install-Package Yolov5Net -Version 1.0.9
```

For CPU usage run this line from Package Manager Console:

```
Install-Package Microsoft.ML.OnnxRuntime -Version 1.9.0
```

For GPU usage run this line from Package Manager Console:

```
Install-Package Microsoft.ML.OnnxRuntime.Gpu -Version 1.9.0
```

CPU and GPU packages can't be installed together.

## Usage

Yolov5Net contains two COCO pre-defined models: YoloCocoP5Model, YoloCocoP6Model. 

If you have custom trained model, then inherit from YoloModel and override all the required properties and methods. See YoloCocoP5Model or YoloCocoP6Model implementation to get know how to wrap your own model. 

```c#
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
```

