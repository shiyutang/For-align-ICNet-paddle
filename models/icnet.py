"""Image Cascade Network"""
import paddle.nn as nn
import paddle.nn.functional as F
import paddle
from segbase import SegBaseModel
from utils import ICNetLoss, SegmentationMetric, SetupLogger


__all__ = ['ICNet']


class ICNet(SegBaseModel):
    """Image Cascade Network"""

    def __init__(self, nclass=19, backbone='resnet50', pretrained_base=True):
        super(ICNet, self).__init__(nclass, backbone, pretrained_base=pretrained_base)
        self.conv_sub1 = nn.Sequential(
            _ConvBNReLU(3, 32, 3, 2),
            _ConvBNReLU(32, 32, 3, 2),
            _ConvBNReLU(32, 64, 3, 2)
        )
        self.ppm = PyramidPoolingModule([1, 2, 3, 6])

        self.head = _ICHead(nclass)

        self.__setattr__('exclusive', ['conv_sub1', 'head'])

    def forward(self, x):
        # sub 1
        x_sub1 = self.conv_sub1(x)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        _, x_sub2, _, _ = self.base_forward(x_sub2)

        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        _, _, _, x_sub4 = self.base_forward(x_sub4)
        # add PyramidPoolingModule
        x_sub4 = self.ppm(x_sub4)

        outputs = self.head(x_sub1, x_sub2, x_sub4)

        return tuple(outputs)


class PyramidPoolingModule(nn.Layer):
    def __init__(self, pyramids):
        super(PyramidPoolingModule, self).__init__()
        if pyramids is None:
            pyramids = [1, 2, 3, 6]
        self.pyramids = pyramids

    def forward(self, input):
        feat = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(x, size=[height, width], mode='bilinear', align_corners=True)
            feat = feat + x
        return feat


class _ICHead(nn.Layer):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2D, **kwargs):
        super(_ICHead, self).__init__()
        # self.cff_12 = CascadeFeatureFusion(512, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, nclass, norm_layer, **kwargs)

        self.conv_cls = nn.Conv2D(128, nclass, 1, bias_attr=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        # x_cff_12, x_12_cls = self.cff_12(x_sub2, x_sub1)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()

        return outputs


# padding卷积里的padding=1对应于torch卷积里padding=2?
class _ConvBNReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2D, bias=False, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CascadeFeatureFusion(nn.Layer):
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2D, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2D(low_channels, out_channels, 3, padding=2, dilation=2, bias_attr=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2D(high_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2D(out_channels, nclass, 1, bias_attr=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


def train_some_iters(model,
                     lr_scheduler,
                     criterion,
                     optimizer,
                     fake_data,
                     fake_label,
                     max_iter=3):
    # needed to avoid network randomness
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', SmoothedValue(
            window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        'img/s', SmoothedValue(
            window_size=10, fmt='{value}'))

    loss_list = []
    lr_list = []
    for idx in range(max_iter):
        image = paddle.to_tensor(fake_data)
        target = paddle.to_tensor(fake_label)

        output = model(image)
        loss = criterion(output, target)

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # print(loss)
        loss_list.append(loss)
        # lr_list.append(optimizer.get_lr())
        lr_list.append(lr_scheduler.get_lr())

    return loss_list, lr_list

if __name__ == '__main__':
    import paddle
    import numpy as np
    from reprod_log import ReprodLogger
    # # To pdparams
    # img = np.load('fake_data.npy', allow_pickle=True)
    # img = paddle.to_tensor(img)
    # model = ICNet()
    # outputs = model(img)
    # print(outputs[3])
    # paddle.save(model.state_dict(), 'paddle_model_init.pdparams')

    # Align
    #
    reprod_logger = ReprodLogger()
    model = ICNet()
    model_file = 'paddle_from_torch.pdparams'
    model.load_dict(paddle.load(model_file))
    model.eval()
    fake_data = np.load('fake_data.npy', allow_pickle=True)
    fake_label = np.load('fake_label.npy', allow_pickle=True)
    input = paddle.to_tensor(fake_data)
    label = paddle.to_tensor(fake_label)
    criterion = ICNetLoss(ignore_index=-1)

    # forward
    # output = model(input)
    # for i in range(4):
    #     reprod_logger.add("forward{}".format(i), output[i].numpy())
    #     reprod_logger.save("forward{}_paddle.npy".format(i))
    #
    # loss = criterion(output, label)
    # print(loss)
    # loss
    # reprod_logger.add("loss", loss.cpu().detach().numpy())
    # reprod_logger.save("loss_paddle.npy")

    # lr
    max_iters = 5
    model.pretrained.parameters().append({'learning_rate': 0.01})
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=0.01,
        decay_steps=30000,
        end_lr=0.0,
        power=0.9,
        cycle=False
    )
    optimizer = paddle.optimizer.Momentum(parameters= model.pretrained.parameters(),
                                               learning_rate=lr_scheduler,
                                               momentum=0.9,
                                               weight_decay=0.0001)
    loss_list, lr_list = train_some_iters(model,
                      lr_scheduler,
                     criterion,
                     optimizer,
                     fake_data,
                     fake_label,
                     max_iter=max_iters)
    # backward
    for i in range(max_iters):
        print(loss_list[i].item())
    for i in range(max_iters):
        reprod_logger.add("loss_backward_{}".format(i), np.array(loss_list[i].item()))
        reprod_logger.save("loss_backward_{}_paddle.npy".format(i))
    # lr
    for i in range(max_iters):
        reprod_logger.add("lr_backward_{}".format(i), np.array(lr_list[i]))
        reprod_logger.save("lr_backward_{}_paddle.npy".format(i))

    # Test
    # inputs = paddle.randn([1, 3, 720, 720])
    # with paddle.no_grad():
    #     outputs = model(inputs)
    # print(len(outputs))		 # 3
    # print(outputs[0].shape) # torch.Size([1, 19, 200, 200])
    # print(outputs[1].shape)# torch.Size([1, 19, 100, 100])
    # print(outputs[2].shape) # torch.Size([1, 19, 50, 50])
    # print(outputs[3].shape) # torch.Size([1, 19, 50, 50])

