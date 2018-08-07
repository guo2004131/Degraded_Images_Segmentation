import fcn
import torch
import PIL.Image
import torchvision
import numpy as np
import os.path as osp
import torch.nn as nn
from collections import namedtuple


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8sDenseGram(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
            path=cls.pretrained_model,
            md5='dbd9bbb3829a3184913bccc74373afbb',
        )

    def __init__(self, n_class=21):
        super(FCN8sDenseGram, self).__init__()

        self.grow_rate = 32
        # conv1 & conv1_
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        self.conv1_1_ = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1_ = nn.ReLU(inplace=True)
        self.conv1_2_ = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2_ = nn.ReLU(inplace=True)
        self.pool1_ = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2 & conv2_
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        self.conv2_1_b = nn.Conv2d(128, self.grow_rate, 1)
        self.relu_2_1_b = nn.ReLU(inplace=True)
        self.conv2_2_b = nn.Conv2d(128 + self.grow_rate, self.grow_rate, 1)
        self.relu_2_2_b = nn.ReLU(inplace=True)

        self.conv2_1_ = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1_ = nn.ReLU(inplace=True)
        self.conv2_2_ = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2_ = nn.ReLU(inplace=True)
        self.pool2_ = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        self.conv2_1_b_ = nn.Conv2d(128, self.grow_rate, 1)
        self.relu_2_1_b_ = nn.ReLU(inplace=True)
        self.conv2_2_b_ = nn.Conv2d(128+self.grow_rate, self.grow_rate, 1)
        self.relu_2_2_b_ = nn.ReLU(inplace=True)

        # conv3 & conv3_
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        self.conv3_1_b = nn.Conv2d(256, self.grow_rate, 1)
        self.relu3_1_b = nn.ReLU(inplace=True)
        self.conv3_2_b = nn.Conv2d(256+self.grow_rate, self.grow_rate, 1)
        self.relu3_2_b = nn.ReLU(inplace=True)
        self.conv3_3_b = nn.Conv2d(256+2*self.grow_rate, self.grow_rate, 1)
        self.relu3_3_b = nn.ReLU(inplace=True)

        self.conv3_1_ = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1_ = nn.ReLU(inplace=True)
        self.conv3_2_ = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2_ = nn.ReLU(inplace=True)
        self.conv3_3_ = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3_ = nn.ReLU(inplace=True)
        self.pool3_ = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        self.conv3_1_b_ = nn.Conv2d(256, self.grow_rate, 1)
        self.relu3_1_b_ = nn.ReLU(inplace=True)
        self.conv3_2_b_ = nn.Conv2d(256+self.grow_rate, self.grow_rate, 1)
        self.relu3_2_b_ = nn.ReLU(inplace=True)
        self.conv3_3_b_ = nn.Conv2d(256+2*self.grow_rate, self.grow_rate, 1)
        self.relu3_3_b_ = nn.ReLU(inplace=True)

        # conv4 & conv4_
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        self.conv4_1_b = nn.Conv2d(512, self.grow_rate, 1)
        self.relu4_1_b = nn.ReLU(inplace=True)
        self.conv4_2_b = nn.Conv2d(512+self.grow_rate, self.grow_rate, 1)
        self.relu4_2_b = nn.ReLU(inplace=True)
        self.conv4_3_b = nn.Conv2d(512+2*self.grow_rate, self.grow_rate, 1)
        self.relu4_3_b = nn.ReLU(inplace=True)

        self.conv4_1_ = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1_ = nn.ReLU(inplace=True)
        self.conv4_2_ = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2_ = nn.ReLU(inplace=True)
        self.conv4_3_ = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3_ = nn.ReLU(inplace=True)
        self.pool4_ = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        self.conv4_1_b_ = nn.Conv2d(512, self.grow_rate, 1)
        self.relu4_1_b_ = nn.ReLU(inplace=True)
        self.conv4_2_b_ = nn.Conv2d(512+self.grow_rate, self.grow_rate, 1)
        self.relu4_2_b_ = nn.ReLU(inplace=True)
        self.conv4_3_b_ = nn.Conv2d(512+2*self.grow_rate, self.grow_rate, 1)
        self.relu4_3_b_ = nn.ReLU(inplace=True)

        # conv5 & conv5_
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        self.conv5_1_b = nn.Conv2d(512, self.grow_rate, 1)
        self.relu5_1_b = nn.ReLU(inplace=True)
        self.conv5_2_b = nn.Conv2d(512+self.grow_rate, self.grow_rate, 1)
        self.relu5_2_b = nn.ReLU(inplace=True)
        self.conv5_3_b = nn.Conv2d(512+2*self.grow_rate, self.grow_rate, 1)
        self.relu5_3_b = nn.ReLU(inplace=True)

        self.conv5_1_ = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1_ = nn.ReLU(inplace=True)
        self.conv5_2_ = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2_ = nn.ReLU(inplace=True)
        self.conv5_3_ = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3_ = nn.ReLU(inplace=True)
        self.pool5_ = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        self.conv5_1_b_ = nn.Conv2d(512, self.grow_rate, 1)
        self.relu5_1_b_ = nn.ReLU(inplace=True)
        self.conv5_2_b_ = nn.Conv2d(512+self.grow_rate, self.grow_rate, 1)
        self.relu5_2_b_ = nn.ReLU(inplace=True)
        self.conv5_3_b_ = nn.Conv2d(512+2*self.grow_rate, self.grow_rate, 1)
        self.relu5_3_b_ = nn.ReLU(inplace=True)

        # fc6 & fc6_
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.fc6_b = nn.Conv2d(4096, self.grow_rate, 1)
        self.relu6_b = nn.ReLU(inplace=True)

        self.fc6_ = nn.Conv2d(512, 4096, 7)
        self.relu6_ = nn.ReLU(inplace=True)
        self.drop6_ = nn.Dropout2d()
        self.fc6_b_ = nn.Conv2d(4096, self.grow_rate, 1)
        self.relu6_b_ = nn.ReLU(inplace=True)

        # fc7 & fc7_
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        self.fc7_b = nn.Conv2d(4096, self.grow_rate, 1)
        self.relu7_b = nn.ReLU(inplace=True)

        self.fc7_ = nn.Conv2d(4096, 4096, 1)
        self.relu7_ = nn.ReLU(inplace=True)
        self.drop7_ = nn.Dropout2d()
        self.fc7_b_ = nn.Conv2d(4096, self.grow_rate, 1)
        self.relu7_b_ = nn.ReLU(inplace=True)

        # transposed conv
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)

        self.score_fr_ = nn.Conv2d(4096, n_class, 1)
        self.score_pool3_ = nn.Conv2d(256, n_class, 1)
        self.score_pool4_ = nn.Conv2d(512, n_class, 1)

        self.upscore2_ = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.upscore8_ = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4_ = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                # torch.nn.init.constant_(m.weight, 1)  # Due to the random dropout, the last output is different
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x, x_):
        h, h_ = x, x_
        # conv1 h
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)
        # conv1 h_
        h_ = self.relu1_1_(self.conv1_1_(h_))
        h_ = self.relu1_2_(self.conv1_2_(h_))
        h_ = self.pool1_(h_)

        # conv2 h
        h = self.relu2_1(self.conv2_1(h))
        f_2_1 = self.relu_2_1_b(self.conv2_1_b(h))
        h = self.relu2_2(self.conv2_2(h))
        f_2_2 = self.relu_2_2_b(self.conv2_2_b(torch.cat([h, f_2_1], 1)))
        h = self.pool2(h)
        # conv2 h_
        h_ = self.relu2_1_(self.conv2_1_(h_))
        f_2_1_ = self.relu_2_1_b_(self.conv2_1_b_(h_))
        h_ = self.relu2_2_(self.conv2_2_(h_))
        f_2_2_ = self.relu_2_2_b_(self.conv2_2_b_(torch.cat([h_, f_2_1_], 1)))
        h_ = self.pool2_(h_)

        # conv3 h
        h = self.relu3_1(self.conv3_1(h))
        f_3_1 = self.relu3_1_b(self.conv3_1_b(h))
        h = self.relu3_2(self.conv3_2(h))
        f_3_2 = self.relu3_2_b(self.conv3_2_b(torch.cat([h, f_3_1], 1)))
        h = self.relu3_3(self.conv3_3(h))
        f_3_3 = self.relu3_3_b(self.conv3_3_b(torch.cat([h, f_3_1, f_3_2], 1)))
        h = self.pool3(h)
        pool3 = h  # 1/8
        # conv3 h_
        h_ = self.relu3_1_(self.conv3_1_(h_))
        f_3_1_ = self.relu3_1_b_(self.conv3_1_b_(h_))
        h_ = self.relu3_2_(self.conv3_2_(h_))
        f_3_2_ = self.relu3_2_b_(self.conv3_2_b_(torch.cat([h_, f_3_1_], 1)))
        h_ = self.relu3_3_(self.conv3_3_(h_))
        f_3_3_ = self.relu3_3_b_(self.conv3_3_b_(torch.cat([h_, f_3_1_, f_3_2_], 1)))
        h_ = self.pool3_(h_)
        pool3_ = h_  # 1/8

        # conv4 h
        h = self.relu4_1(self.conv4_1(h))
        f_4_1 = self.relu4_1_b(self.conv4_1_b(h))
        h = self.relu4_2(self.conv4_2(h))
        f_4_2 = self.relu4_2_b(self.conv4_2_b(torch.cat([h, f_4_1], 1)))
        h = self.relu4_3(self.conv4_3(h))
        f_4_3 = self.relu4_3_b(self.conv4_3_b(torch.cat([h, f_4_1, f_4_2], 1)))
        h = self.pool4(h)
        pool4 = h  # 1/16
        # conv4 h_
        h_ = self.relu4_1_(self.conv4_1_(h_))
        f_4_1_ = self.relu4_1_b_(self.conv4_1_b_(h_))
        h_ = self.relu4_2_(self.conv4_2_(h_))
        f_4_2_ = self.relu4_2_b_(self.conv4_2_b_(torch.cat([h_, f_4_1_], 1)))
        h_ = self.relu4_3_(self.conv4_3_(h_))
        f_4_3_ = self.relu4_3_b_(self.conv4_3_b_(torch.cat([h_, f_4_1_, f_4_2_], 1)))
        h_ = self.pool4_(h_)
        pool4_ = h_  # 1/16

        # conv5 h
        h = self.relu5_1(self.conv5_1(h))
        f_5_1 = self.relu5_1_b(self.conv5_1_b(h))
        h = self.relu5_2(self.conv5_2(h))
        f_5_2 = self.relu5_2_b(self.conv5_2_b(torch.cat([h, f_5_1], 1)))
        h = self.relu5_3(self.conv5_3(h))
        f_5_3 = self.relu5_3_b(self.conv5_3_b(torch.cat([h, f_5_1, f_5_2], 1)))
        h = self.pool5(h)
        # conv5 h_
        h_ = self.relu5_1_(self.conv5_1_(h_))
        f_5_1_ = self.relu5_1_b_(self.conv5_1_b_(h_))
        h_ = self.relu5_2_(self.conv5_2_(h_))
        f_5_2_ = self.relu5_2_b_(self.conv5_2_b_(torch.cat([h_, f_5_1_], 1)))
        h_ = self.relu5_3_(self.conv5_3_(h_))
        f_5_3_ = self.relu5_3_b_(self.conv5_3_b_(torch.cat([h_, f_5_1_, f_5_2_], 1)))
        h_ = self.pool5_(h_)

        # fc6 h
        h = self.relu6(self.fc6(h))
        f_6 = self.relu6_b(self.fc6_b(h))
        h = self.drop6(h)
        # fc6 h_
        h_ = self.relu6_(self.fc6_(h_))
        f_6_ = self.relu6_b_(self.fc6_b_(h_))
        h_ = self.drop6_(h_)

        # fc7 h
        h = self.relu7(self.fc7(h))
        f_7 = self.relu7_b(self.fc7_b(h))
        h = self.drop7(h)
        # fc7 h_
        h_ = self.relu7_(self.fc7_(h_))
        f_7_ = self.relu7_b_(self.fc7_b_(h_))
        h_ = self.drop7_(h_)

        # upscore2 h
        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16
        # upscore2 h_
        h_ = self.score_fr_(h_)
        h_ = self.upscore2_(h_)
        upscore2_ = h_  # 1/16

        # score_pool4c h
        h = self.score_pool4(pool4)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        # score_pool4c h_
        h_ = self.score_pool4_(pool4_)  # XXX: scaling to train at once
        h_ = h_[:, :, 5:5 + upscore2_.size()[2], 5:5 + upscore2_.size()[3]]
        score_pool4c_ = h_  # 1/16

        # upscore_pool4 h
        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        # upscore_pool4 h_
        h_ = upscore2_ + score_pool4c_  # 1/16
        h_ = self.upscore_pool4_(h_)
        upscore_pool4_ = h_  # 1/8

        # score_pool3c h
        h = self.score_pool3(pool3)  # XXX: scaling to train at once
        h = h[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        # score_pool3c h_
        h_ = self.score_pool3_(pool3_)  # XXX: scaling to train at once
        h_ = h_[:, :, 9:9 + upscore_pool4_.size()[2], 9:9 + upscore_pool4_.size()[3]]
        score_pool3c_ = h_  # 1/8

        # merge pool4 and pool3c h
        h = upscore_pool4 + score_pool3c  # 1/8
        # merge pool4 and pool3c h_
        h_ = upscore_pool4_ + score_pool3c_  # 1/8

        # merge all h
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        # merge all h_
        h_ = self.upscore8_(h_)
        h_ = h_[:, :, 31:31 + x_.size()[2], 31:31 + x_.size()[3]].contiguous()

        gram_outputs = namedtuple("GramOutputs", ['f_2_1', 'f_2_2',
                                                  'f_3_1', 'f_3_2', 'f_3_3',
                                                  'f_4_1', 'f_4_2', 'f_4_3',
                                                  'f_5_1', 'f_5_2', 'f_5_3',
                                                  'f_6', 'f_7'])
        gram_outputs_ = namedtuple("GramOutputs_", ['f_2_1_', 'f_2_2_',
                                                    'f_3_1_', 'f_3_2_', 'f_3_3_',
                                                    'f_4_1_', 'f_4_2_', 'f_4_3_',
                                                    'f_5_1_', 'f_5_2_', 'f_5_3_',
                                                    'f_6_', 'f_7_'])
        gouts = gram_outputs(f_2_1, f_2_2,
                             f_3_1, f_3_2, f_3_3,
                             f_4_1, f_4_2, f_4_3,
                             f_5_1, f_5_2, f_5_3,
                             f_6, f_7)
        gouts_ = gram_outputs_(f_2_1_, f_2_2_,
                               f_3_1_, f_3_2_, f_3_3_,
                               f_4_1_, f_4_2_, f_4_3_,
                               f_5_1_, f_5_2_, f_5_3_,
                               f_6_, f_7_)
        return h, h_, gouts, gouts_

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)


class FCN8sAtOnceDenseGram(FCN8sDenseGram):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s-atonce_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
            path=cls.pretrained_model,
            md5='bfed4437e941fef58932891217fe6464',
        )

    def forward(self, x, x_):
        h, h_ = x, x_
        # conv1 h
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)
        # conv1 h_
        h_ = self.relu1_1_(self.conv1_1_(h_))
        h_ = self.relu1_2_(self.conv1_2_(h_))
        h_ = self.pool1_(h_)

        # conv2 h
        h = self.relu2_1(self.conv2_1(h))
        f_2_1 = self.relu_2_1_b(self.conv2_1_b(h))
        h = self.relu2_2(self.conv2_2(h))
        f_2_2 = self.relu_2_2_b(self.conv2_2_b(torch.cat([h, f_2_1], 1)))
        h = self.pool2(h)
        # conv2 h_
        h_ = self.relu2_1_(self.conv2_1_(h_))
        f_2_1_ = self.relu_2_1_b_(self.conv2_1_b_(h_))
        h_ = self.relu2_2_(self.conv2_2_(h_))
        f_2_2_ = self.relu_2_2_b_(self.conv2_2_b_(torch.cat([h_, f_2_1_], 1)))
        h_ = self.pool2_(h_)

        # conv3 h
        h = self.relu3_1(self.conv3_1(h))
        f_3_1 = self.relu3_1_b(self.conv3_1_b(h))
        h = self.relu3_2(self.conv3_2(h))
        f_3_2 = self.relu3_2_b(self.conv3_2_b(torch.cat([h, f_3_1], 1)))
        h = self.relu3_3(self.conv3_3(h))
        f_3_3 = self.relu3_3_b(self.conv3_3_b(torch.cat([h, f_3_1, f_3_2], 1)))
        h = self.pool3(h)
        pool3 = h  # 1/8
        # conv3 h_
        h_ = self.relu3_1_(self.conv3_1_(h_))
        f_3_1_ = self.relu3_1_b_(self.conv3_1_b_(h_))
        h_ = self.relu3_2_(self.conv3_2_(h_))
        f_3_2_ = self.relu3_2_b_(self.conv3_2_b_(torch.cat([h_, f_3_1_], 1)))
        h_ = self.relu3_3_(self.conv3_3_(h_))
        f_3_3_ = self.relu3_3_b_(self.conv3_3_b_(torch.cat([h_, f_3_1_, f_3_2_], 1)))
        h_ = self.pool3_(h_)
        pool3_ = h_  # 1/8

        # conv4 h
        h = self.relu4_1(self.conv4_1(h))
        f_4_1 = self.relu4_1_b(self.conv4_1_b(h))
        h = self.relu4_2(self.conv4_2(h))
        f_4_2 = self.relu4_2_b(self.conv4_2_b(torch.cat([h, f_4_1], 1)))
        h = self.relu4_3(self.conv4_3(h))
        f_4_3 = self.relu4_3_b(self.conv4_3_b(torch.cat([h, f_4_1, f_4_2], 1)))
        h = self.pool4(h)
        pool4 = h  # 1/16
        # conv4 h_
        h_ = self.relu4_1_(self.conv4_1_(h_))
        f_4_1_ = self.relu4_1_b_(self.conv4_1_b_(h_))
        h_ = self.relu4_2_(self.conv4_2_(h_))
        f_4_2_ = self.relu4_2_b_(self.conv4_2_b_(torch.cat([h_, f_4_1_], 1)))
        h_ = self.relu4_3_(self.conv4_3_(h_))
        f_4_3_ = self.relu4_3_b_(self.conv4_3_b_(torch.cat([h_, f_4_1_, f_4_2_], 1)))
        h_ = self.pool4_(h_)
        pool4_ = h_  # 1/16

        # conv5 h
        h = self.relu5_1(self.conv5_1(h))
        f_5_1 = self.relu5_1_b(self.conv5_1_b(h))
        h = self.relu5_2(self.conv5_2(h))
        f_5_2 = self.relu5_2_b(self.conv5_2_b(torch.cat([h, f_5_1], 1)))
        h = self.relu5_3(self.conv5_3(h))
        f_5_3 = self.relu5_3_b(self.conv5_3_b(torch.cat([h, f_5_1, f_5_2], 1)))
        h = self.pool5(h)
        # conv5 h_
        h_ = self.relu5_1_(self.conv5_1_(h_))
        f_5_1_ = self.relu5_1_b_(self.conv5_1_b_(h_))
        h_ = self.relu5_2_(self.conv5_2_(h_))
        f_5_2_ = self.relu5_2_b_(self.conv5_2_b_(torch.cat([h_, f_5_1_], 1)))
        h_ = self.relu5_3_(self.conv5_3_(h_))
        f_5_3_ = self.relu5_3_b_(self.conv5_3_b_(torch.cat([h_, f_5_1_, f_5_2_], 1)))
        h_ = self.pool5_(h_)

        # fc6 h
        h = self.relu6(self.fc6(h))
        f_6 = self.relu6_b(self.fc6_b(h))
        h = self.drop6(h)
        # fc6 h_
        h_ = self.relu6_(self.fc6_(h_))
        f_6_ = self.relu6_b_(self.fc6_b_(h_))
        h_ = self.drop6_(h_)

        # fc7 h
        h = self.relu7(self.fc7(h))
        f_7 = self.relu7_b(self.fc7_b(h))
        h = self.drop7(h)
        # fc7 h_
        h_ = self.relu7_(self.fc7_(h_))
        f_7_ = self.relu7_b_(self.fc7_b_(h_))
        h_ = self.drop7_(h_)

        # upscore2 h
        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16
        # upscore2 h_
        h_ = self.score_fr_(h_)
        h_ = self.upscore2_(h_)
        upscore2_ = h_  # 1/16

        # score_pool4c h
        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        # score_pool4c h_
        h_ = self.score_pool4_(pool4_ * 0.01)  # XXX: scaling to train at once
        h_ = h_[:, :, 5:5 + upscore2_.size()[2], 5:5 + upscore2_.size()[3]]
        score_pool4c_ = h_  # 1/16

        # upscore_pool4 h
        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        # upscore_pool4 h_
        h_ = upscore2_ + score_pool4c_  # 1/16
        h_ = self.upscore_pool4_(h_)
        upscore_pool4_ = h_  # 1/8

        # score_pool3c h
        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        # score_pool3c h_
        h_ = self.score_pool3_(pool3_ * 0.0001)  # XXX: scaling to train at once
        h_ = h_[:, :, 9:9 + upscore_pool4_.size()[2], 9:9 + upscore_pool4_.size()[3]]
        score_pool3c_ = h_  # 1/8

        # merge pool4 and pool3c h
        h = upscore_pool4 + score_pool3c  # 1/8
        # merge pool4 and pool3c h_
        h_ = upscore_pool4_ + score_pool3c_  # 1/8

        # merge all h
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        # merge all h_
        h_ = self.upscore8_(h_)
        h_ = h_[:, :, 31:31 + x_.size()[2], 31:31 + x_.size()[3]].contiguous()

        gram_outputs = namedtuple("GramOutputs", ['f_2_1', 'f_2_2',
                                                  'f_3_1', 'f_3_2', 'f_3_3',
                                                  'f_4_1', 'f_4_2', 'f_4_3',
                                                  'f_5_1', 'f_5_2', 'f_5_3',
                                                  'f_6', 'f_7'])
        gram_outputs_ = namedtuple("GramOutputs_", ['f_2_1_', 'f_2_2_',
                                                    'f_3_1_', 'f_3_2_', 'f_3_3_',
                                                    'f_4_1_', 'f_4_2_', 'f_4_3_',
                                                    'f_5_1_', 'f_5_2_', 'f_5_3_',
                                                    'f_6_', 'f_7_'])
        gouts = gram_outputs(f_2_1, f_2_2,
                             f_3_1, f_3_2, f_3_3,
                             f_4_1, f_4_2, f_4_3,
                             f_5_1, f_5_2, f_5_3,
                             f_6, f_7)
        gouts_ = gram_outputs_(f_2_1_, f_2_2_,
                               f_3_1_, f_3_2_, f_3_3_,
                               f_4_1_, f_4_2_, f_4_3_,
                               f_5_1_, f_5_2_, f_5_3_,
                               f_6_, f_7_)
        return h, h_, gouts, gouts_

    def copy_params_from_vgg16_fcn8s(self, vgg16, fcn8s):
        fcn8snames = [
            'conv1_1', 'conv1_2',
            'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3',
            'conv4_1', 'conv4_2', 'conv4_3',
            'conv5_1', 'conv5_2', 'conv5_3',
            'fc6', 'fc7',
            'score_fr',
            'score_pool3', 'score_pool4',
            'upscore2', 'upscore8',
            'upscore_pool4'
        ]
        features = [
            self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2, self.conv3_3,
            self.conv4_1, self.conv4_2, self.conv4_3,
            self.conv5_1, self.conv5_2, self.conv5_3,
            self.fc6, self.fc7,
            self.score_fr,
            self.score_pool3, self.score_pool4,
            self.upscore2, self.upscore8,
            self.upscore_pool4
        ]

        features_ = [
            self.conv1_1_, self.relu1_1_,
            self.conv1_2_, self.relu1_2_,
            self.pool1,
            self.conv2_1_, self.relu2_1_,
            self.conv2_2_, self.relu2_2_,
            self.pool2_,
            self.conv3_1_, self.relu3_1_,
            self.conv3_2_, self.relu3_2_,
            self.conv3_3_, self.relu3_3_,
            self.pool3_,
            self.conv4_1_, self.relu4_1_,
            self.conv4_2_, self.relu4_2_,
            self.conv4_3_, self.relu4_3_,
            self.pool4_,
            self.conv5_1_, self.relu5_1_,
            self.conv5_2_, self.relu5_2_,
            self.conv5_3_, self.relu5_3_,
            self.pool5_,
        ]
        # features -- teacher (fcn8s)
        for l1, l2 in zip(fcn8snames, features):
            l1_weight = l1 + '.weight'
            l1_bias = l1 + '.bias'
            if isinstance(l2, nn.Conv2d):
                assert fcn8s[l1_weight].size() == l2.weight.size()
                assert fcn8s[l1_bias].size() == l2.bias.size()
                l2.weight.data.copy_(fcn8s[l1_weight])
                l2.bias.data.copy_(fcn8s[l1_bias])
            if isinstance(l2, nn.ConvTranspose2d):
                assert fcn8s[l1_weight].size() == l2.weight.size()
                l2.weight.data.copy_(fcn8s[l1_weight])
        # features_ -- student (vgg16)
        for l1, l2 in zip(vgg16.features, features_):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6_', 'fc7_']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

    def copy_params_from_fcn8s(self, fcn8s):
        fcn8snames = [
            'conv1_1', 'conv1_2',
            'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3',
            'conv4_1', 'conv4_2', 'conv4_3',
            'conv5_1', 'conv5_2', 'conv5_3',
            'fc6', 'fc7',
            'score_fr',
            'score_pool3', 'score_pool4',
            'upscore2', 'upscore8',
            'upscore_pool4',
        ]
        features = [
            self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2, self.conv3_3,
            self.conv4_1, self.conv4_2, self.conv4_3,
            self.conv5_1, self.conv5_2, self.conv5_3,
            self.fc6, self.fc7,
            self.score_fr,
            self.score_pool3, self.score_pool4,
            self.upscore2, self.upscore8,
            self.upscore_pool4,
        ]

        features_ = [
            self.conv1_1_, self.conv1_2_,
            self.conv2_1_, self.conv2_2_,
            self.conv3_1_, self.conv3_2_, self.conv3_3_,
            self.conv4_1_, self.conv4_2_, self.conv4_3_,
            self.conv5_1_, self.conv5_2_, self.conv5_3_,
            self.fc6_, self.fc7_,
            self.score_fr_,
            self.score_pool3_, self.score_pool4_,
            self.upscore2_, self.upscore8_,
            self.upscore_pool4_,
        ]
        # features -- teacher (fcn8s)
        for l1, l2 in zip(fcn8snames, features):
            l1_weight = l1 + '.weight'
            l1_bias = l1 + '.bias'
            if isinstance(l2, nn.Conv2d):
                assert fcn8s[l1_weight].size() == l2.weight.size()
                assert fcn8s[l1_bias].size() == l2.bias.size()
                l2.weight.data.copy_(fcn8s[l1_weight])
                l2.bias.data.copy_(fcn8s[l1_bias])
            if isinstance(l2, nn.ConvTranspose2d):
                assert fcn8s[l1_weight].size() == l2.weight.size()
                l2.weight.data.copy_(fcn8s[l1_weight])
        # features_ -- student (fcn8s)
        for l1, l2 in zip(fcn8snames, features_):
            l1_weight = l1 + '.weight'
            l1_bias = l1 + '.bias'
            if isinstance(l2, nn.Conv2d):
                assert fcn8s[l1_weight].size() == l2.weight.size()
                assert fcn8s[l1_bias].size() == l2.bias.size()
                l2.weight.data.copy_(fcn8s[l1_weight])
                l2.bias.data.copy_(fcn8s[l1_bias])
            if isinstance(l2, nn.ConvTranspose2d):
                assert fcn8s[l1_weight].size() == l2.weight.size()
                l2.weight.data.copy_(fcn8s[l1_weight])

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
            self.fc6, self.fc7,
            self.score_fr,
            self.score_pool3, self.score_pool4,
            self.upscore2, self.upscore8,
            self.upscore_pool4
        ]

        features_ = [
            self.conv1_1_, self.relu1_1_,
            self.conv1_2_, self.relu1_2_,
            self.pool1,
            self.conv2_1_, self.relu2_1_,
            self.conv2_2_, self.relu2_2_,
            self.pool2_,
            self.conv3_1_, self.relu3_1_,
            self.conv3_2_, self.relu3_2_,
            self.conv3_3_, self.relu3_3_,
            self.pool3_,
            self.conv4_1_, self.relu4_1_,
            self.conv4_2_, self.relu4_2_,
            self.conv4_3_, self.relu4_3_,
            self.pool4_,
            self.conv5_1_, self.relu5_1_,
            self.conv5_2_, self.relu5_2_,
            self.conv5_3_, self.relu5_3_,
            self.pool5_,
            self.fc6_, self.fc7_,
            self.score_fr_,
            self.score_pool3_, self.score_pool4_,
            self.upscore2_, self.upscore8_,
            self.upscore_pool4_
        ]
        # features -- teacher (vgg16)
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
        # features_ -- student (vgg16)
        for l1, l2 in zip(vgg16.features, features_):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6_', 'fc7_']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

    def freeze_params(self, t, s):
        """
        freeze network paramters
        :param t: freeze teacher network parameters 0; update teacher network parameters (NOT 0)
        :param s: freeze student network parameters 0; update student network parameters (NOT 0)
        :return: N/A
        """
        features = [
            self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2, self.conv3_3,
            self.conv4_1, self.conv4_2, self.conv4_3,
            self.conv5_1, self.conv5_2, self.conv5_3,
            self.fc6, self.fc7,
            self.score_fr,
            self.score_pool3, self.score_pool4,
            self.upscore2, self.upscore8,
            self.upscore_pool4,
        ]
        features_ = [
            self.conv1_1_, self.conv1_2_,
            self.conv2_1_, self.conv2_2_,
            self.conv3_1_, self.conv3_2_, self.conv3_3_,
            self.conv4_1_, self.conv4_2_, self.conv4_3_,
            self.conv5_1_, self.conv5_2_, self.conv5_3_,
            self.fc6_, self.fc7_,
            self.score_fr_,
            self.score_pool3_, self.score_pool4_,
            self.upscore2_, self.upscore8_,
            self.upscore_pool4_,
        ]
        if t == 0:
            for l in features:
                l.weight.requires_grad = False
        else:
            for l in features:
                l.weight.requires_grad = True
        if s == 0:
            for l in features_:
                l.weight.requires_grad = False
        else:
            for l in features_:
                l.weight.requires_grad = True


# for code testing & debugging
if __name__ == "__main__":
    fcn8satoncedg = FCN8sAtOnceDenseGram(n_class=12)
    model_fcn8s = torch.load('/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-e7d2d0f_TIME-20180804-134817/model_best.pth.tar')
    model_fcn8s = model_fcn8s['model_state_dict']
    model_vgg16 = torchvision.models.vgg16(pretrained=True)
    # fcn8satoncedg.copy_params_from_vgg16_fcn8s(model_vgg16, model_fcn8s)
    fcn8satoncedg.copy_params_from_fcn8s(model_fcn8s)
    # fcn8satoncedg.copy_params_from_vgg16(model_vgg16)

    # test the network when given two inputs
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    img0 = PIL.Image.open('/home/dg/Dropbox/Datasets/CamVid/Original_Images/CamVid_test_images/0001TP_008550.png')
    img1 = PIL.Image.open('/home/dg/Dropbox/Datasets/CamVid/Degraded_Images/Haze/degraded_parameter_0.5/CamVid_test_images/0001TP_008550.png')
    img0 = np.array(img0).astype(np.float64)
    img1 = np.array(img1).astype(np.float64)

    img0 = img0[:, :, ::-1]
    img0 -= mean_bgr
    img0 = img0.transpose(2, 0, 1)
    img0 = torch.from_numpy(img0.copy()).float()
    img0 = img0.view(1, 3, 360, 480)

    img1 = img1[:, :, ::-1]
    img1 -= mean_bgr
    img1 = img1.transpose(2, 0, 1)
    img1 = torch.from_numpy(img1.copy()).float()
    img1 = img1.view(1, 3, 360, 480)

    fcn8satoncedg = fcn8satoncedg.cuda(device=0)
    img0 = img0.cuda(device=0)
    img1 = img1.cuda(device=0)
    h, h_, gramouts, gramouts_ = fcn8satoncedg(img0, img1)

    # h, h_, gramouts, gramouts_ = fcn8satoncedg(img0, img0)  # debug the network.
    print gramouts.f_5_3.shape
    print gramouts_.f_5_3_.shape

    p = h.data.max(1)[1].cpu().numpy()[:, :, :].astype(np.int8)
    p_ = h_.data.max(1)[1].cpu().numpy()[:, :, :].astype(np.int8)
    import matplotlib.pyplot as plt
    plt.imshow(p[0, :, :])
    plt.show()
    plt.imshow(p_[0, :, :])
    plt.show()

    # test gram matrix & perceptual loss for f_5_1 and f_5_1_
    import utils
    import torch.nn.functional as F
    for l1, l2 in zip(gramouts, gramouts_):
        g1 = utils.gram_matrix(l1)
        g2 = utils.gram_matrix(l2)
        loss = F.mse_loss(g1, g2)
        print loss
    print

    # test freeze parameters
    fcn8satoncedg.freeze_params(0, 1)
    print fcn8satoncedg.conv1_1.weight.requires_grad
    print fcn8satoncedg.conv1_1_.weight.requires_grad

