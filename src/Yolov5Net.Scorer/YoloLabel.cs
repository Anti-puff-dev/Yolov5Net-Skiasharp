using SkiaSharp;
using System.Drawing;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Label of detected object.
    /// </summary>
    public class YoloLabel
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public YoloLabelKind Kind { get; set; }
        public SKColor Color { get; set; }

        public YoloLabel()
        {
            Color = SKColor.Parse("FFFF00");
        }
    }
}
