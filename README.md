# Description
This repo contains ICNet implemented by Paddlepaddle2.2.0, based on [paper](https://arxiv.org/abs/1704.08545) by Hengshuang Zhao, and et. al(ECCV'18).
Training and evaluation are done on the [Cityscapes dataset](https://www.cityscapes-dataset.com/) by default.

# Requirements
Python 3.7 or later with the following `pip3 install -r requirements.txt`:
- paddlepaddle-gpu==2.2.0rc0
- numpy==1.17.0
- Pillow==6.0.0
- PyYAML==5.1.2

# Align
- All the logs and related files can be found in the `align/`, including **forward, metric, loss and backward**.
- Get [icnet_resnet50_197_0.710_best_model.pth]()

# Performance  
- Base on Cityscapes dataset, only train on trainning set, and test on validation set, using only one Tesla V100 card on [aistudio platform](https://aistudio.baidu.com/aistudio/index), and input size of the test phase is 2048x1024x3.

|    Method     | mIoU(%) | Memory(GB) |    GPU     | Time(train) |
| :-----------: | :-----: | :--------: | :--------: | :---------: |
| ICNet(paper)  |  67.7%  |  **1.6**   |   TitanX   |    80+h     |
| ICNet(torch)  |  71.0%  |    1.86    | GTX 1080Ti |     50h     |
| ICNet(paddle) |         |    2.85    | Tesla V100 |   **15h**   |



# Demo

|image|predict|
|:---:|:---:|
|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/frankfurt_000001_057181_leftImg8bit_src.png)|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/frankfurt_000001_057181_leftImg8bit_mIoU_0.727.png)|
|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/lindau_000005_000019_leftImg8bit_src.png)|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/lindau_000005_000019_leftImg8bit_mIoU_0.705.png) |
|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000061_000019_leftImg8bit_src.png)|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000061_000019_leftImg8bit_mIoU_0.704.png) |
|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000121_000019_leftImg8bit_src.png)|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000121_000019_leftImg8bit_mIoU_0.694.png) |
|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000124_000019_leftImg8bit_src.png)|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000124_000019_leftImg8bit_mIoU_0.696.png) |
|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000150_000019_leftImg8bit_src.png)|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000150_000019_leftImg8bit_mIoU_0.696.png) |
|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000158_000019_leftImg8bit_src.png)|![](/Users/liangshihao01/Desktop/ICNet/ICNet-paddle/demo/munster_000158_000019_leftImg8bit_mIoU_0.692.png) |
|- All the input images comes from the validation dataset of the Cityscaps, you can switch to the `demo/` directory to check more demo results.||

# Usage
## Preparation
pretrained r50 models：

- [torch](https://pan.baidu.com/s/1VlERTfXbapp9LO4Vlj8CGg): code is wpbb 
- [paddle](https://pan.baidu.com/s/1MfdpOEz7XKXF2Z4fZxJykg) (transformed from torch): code is 27zy 

ICNet best models：

- [torch](https://pan.baidu.com/s/1CFP_c2Hr_HqTkxGwiYe-JA): code is 39ut
- [paddle](https://pan.baidu.com/s/1MfdpOEz7XKXF2Z4fZxJykg) (reproduction): code is 


## Trainning

First, modify the configuration in the `configs/icnet.yaml` file:
```Python
### Trainning 
train:
  specific_gpu_num: "1"   # for example: "0", "1" or "0, 1"
  train_batch_size: 7    # adjust according to gpu resources
  cityscapes_root: "./data/Cityscapes/" 
  ckpt_dir: "./ckpt/"     # ckpt and trainning log will be saved here
```
Then, run: `python3 train.py`

## Evaluation
First, modify the configuration in the `configs/icnet.yaml` file:
```Python
### Test
test:
  ckpt_path: "./ckpt/icnet_resnet50_197_0.710_best_model.pdparams"  # set the pretrained model path correctly
```
Then, run: `python3 evaluate.py`



# Discussion
#### Network Structure

The structure of ICNet is mainly composed of `sub4`, `sub2`, `sub1` and `head`: 

- `sub4`: basically a `pspnet`, the biggest difference is a modified `pyramid pooling module`.

- `sub2`: the first three phases convolutional layers of `sub4`, `sub2` and `sub4` share these three phases convolutional layers.

- `sub1`: three consecutive stried convolutional layers, to fastly downsample the original large-size input images

- `head`: through the `CFF` module, the outputs of the three cascaded branches( `sub4`, `sub2` and `sub1`) are connected. Finaly, using 1x1 convolution and interpolation to get the output.

#### Issues

- During the training, I found some issues. **Paddlepaddle-2.1.2** does not support constructing optimizers which can specify sublayers' learning rates or other parameters. After updating to **paddlepaddle-2.2.0rc**, the problem is solved. 
- 

#### Tricks

Data preprocessing： set the `crop_size` as close as possible to the input size of prediction phase. Here are some experiments from [liminn-ICNet-pytorch](https://github.com/liminn/ICNet-pytorch) :

- `base_size` to **520**, it means resize the shorter side of image between 520x0.5 and 520x2, and set the `crop size` to **480**, it means randomly crop 480x480 patch to train. The final best mIoU is **66.7%**.
- `base_size` to **1024**, it means resize the shorter side of image between 1024x0.5 and 1024x2, and set the `crop_size` to **720**, it means randomly crop 720x720 patch to train. The final best mIoU is **69.9%**.
- Beacuse the target dataset is Cityscapes, the image size is 2048x1024, so a large `crop_size`( 960x960 ) is better. It is believed that larger `crop_size` will bring higher mIoU, such as `crop_size `960, the mIoU is ____%. But large `crop_size` ( such as 1024x1024 ) will result in a smaller batch size and is very time-consuming. 

- set the learning rate of `sub4` to orginal initial learning rate(**0.01**), because it has backbone pretrained weights.
- set the learning rate of `sub1` and `head` to 10 times initial learning rate(**0.1**), because there are no pretrained weights for them.

#### Further works

For experiments in paddle, there are further jobs to do, such as using different `crop_size` 720, 1024 to see how data preprocessing influences model's performance. 

# Reference
- [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)
- [liminn-ICNet-pytorch](https://github.com/liminn/ICNet-pytorch)
- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- [Human-Segmentation-PyTorch](https://github.com/thuyngch/Human-Segmentation-PyTorch)