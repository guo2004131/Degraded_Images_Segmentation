import fcn
import torch
import os.path as osp
import torch.nn as nn
from collections import namedtuple
from .fcn32s import get_upsampling_weight


class FCN8s(nn.Module):

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
        super(FCN8s, self).__init__()

        self.grow_rate = 32
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

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
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x, x_):
        h, h_ = x, x_
        # conv1 -- do not change
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

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
        f_3_3 = self.relu_3_3_b(self.conv3_3_b(torch.cat([h, f_3_1, f_3_2], 1)))
        h = self.pool3(h)
        pool3 = h  # 1/8
        # conv3 h_
        h_ = self.relu3_1_(self.conv3_1_(h_))
        f_3_1_ = self.relu3_1_b_(self.conv3_1_b_(h_))
        h_ = self.relu3_2_(self.conv3_2_(h_))
        f_3_2_ = self.relu3_2_b_(self.conv3_2_b_(torch.cat([h_, f_3_1_], 1)))
        h_ = self.relu3_3_(self.conv3_3_(h_))
        f_3_3_ = self.relu_3_3_b_(self.conv3_3_b_(torch.cat([h_, f_3_1_, f_3_2_], 1)))
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
        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        # score_pool4c h_
        h_ = self.score_pool4_(pool4_)
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
        h = self.score_pool3(pool3)
        h = h[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        # score_pool3c h_
        h_ = self.score_pool3_(pool3_)
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
        h_ = self.upscore8(h_)
        h_ = h_[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        gram_outputs = namedtuple("GramOutputs", ['f_2_1', 'f_2_2', 'f_2_1_', 'f_2_2_',
                                                  'f_3_1', 'f_3_2', 'f_3_3', 'f_3_1_', 'f_3_2_', 'f_3_3_',
                                                  'f_4_1', 'f_4_2', 'f_4_3', 'f_4_1_', 'f_4_2_', 'f_4_3_',
                                                  'f_5_1', 'f_5_2', 'f_5_3', 'f_5_1_', 'f_5_2_', 'f_5_3_',
                                                  'f_6', 'f_6_', 'f_7', 'f_7_'])
        gouts = gram_outputs(f_2_1, f_2_2, f_2_1_, f_2_2_,
                             f_3_1, f_3_2, f_3_3, f_3_1_, f_3_2_, f_3_3_,
                             f_4_1, f_4_2, f_4_3, f_4_1_, f_4_2_, f_4_3_,
                             f_5_1, f_5_2, f_5_3, f_5_1_, f_5_2_, f_5_3_,
                             f_6, f_6_, f_7, f_7_)

        return h, h_, gouts

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


class FCN8sAtOnce(FCN8s):

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
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

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
        ]
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
