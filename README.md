# U-Net-in-PyTorch
This project implements the U-Net architecture in PyTorch for semantic segmentation tasks. U-Net is widely used in medical image segmentation and other pixel-wise classification problems.

## Features
-Encoder-decoder architecture with skip connections
-Custom modules: DoubleConv, DownSample, UpSample
-Fully convolutional network (FCN)
-Easily extendable for multi-class segmentation

## Architecture
``` text
Input (3 x 512 x 512)
  ↓
[DownSample] → [DownSample] → [DownSample] → [DownSample]
  ↓              ↓              ↓              ↓
               Bottleneck (1024)
  ↑              ↑              ↑              ↑
[UpSample] ← [UpSample] ← [UpSample] ← [UpSample]
  ↓
Output (1 x 512 x 512)
```
