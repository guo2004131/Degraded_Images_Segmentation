import os
import fcn
import math
import pytz
import tqdm
import torch
import utils
import shutil
import datetime
import scipy.misc
import numpy as np
import os.path as osp
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable


class TrainerDG(object):

    def __init__(self, cuda, model, optimizer, train_loader, val_loader, test_loader, out, max_iter,
                 size_average=False, interval_validate=None):

        self.cuda = cuda

        self.model = model
        self.optim = optimizer
        self.init_lr = optimizer.param_groups[0]['lr']

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.timestamp_start = datetime.datetime.now(pytz.timezone('America/New_York'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.ignore = self.train_loader.dataset.class_ignore
        self.max_iter = max_iter
        self.best_mean_iu = 0
        self.best_train_meanIoU = 0

        self.t_logger = utils.Logger(self.out, 'train')
        self.v_logger = utils.Logger(self.out, 'valid')
        self.ts_logger = utils.Logger(self.out, 'test')

    def test(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.test_loader.dataset.class_names)

        test_loss = 0
        test_loss_seg = 0
        test_loss_gram = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data_o, data_d, target) in tqdm.tqdm(
                enumerate(self.test_loader), total=len(self.test_loader),
                desc='Test iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data_o, data_d, target = data_o.cuda(), data_d.cuda(), target.cuda()
            data_o, data_d, target = Variable(data_o), Variable(data_d), Variable(target)
            score, score_, gram, gram_ = self.model(data_o, data_d)

            # segmentation loss
            loss_seg = utils.cross_entropy2d(score_, target, size_average=self.size_average)
            # gram loss
            loss_gram = 0
            for g, g_ in zip(gram, gram_):
                loss_gram += F.mse_loss(utils.gram_matrix(g), utils.gram_matrix(g_))
            # loss_gram /= len(gram)
            # overall loss
            loss = loss_seg + loss_gram
            # loss data
            loss_data = float(loss.data[0])
            loss_seg_data = float(loss_seg.data[0])
            loss_gram_data = float(loss_gram.data[0])

            if np.isnan(loss_data):
                raise ValueError('loss is nan while testing')
            test_loss += loss_data / len(data_o)
            test_loss_seg += loss_seg_data / len(data_o)
            test_loss_gram += loss_gram_data / len(data_o)

            imgs_o = data_o.data.cpu()
            imgs_d = data_d.data.cpu()
            # predition from the student network -- score_
            lbl_pred = score_.data.max(1)[1].cpu().numpy()[:, :, :].astype(np.int8)
            lbl_true = target.data.cpu().numpy()
            for img_o, img_d, lt, lp in zip(imgs_o, imgs_d, lbl_true, lbl_pred):
                img_o, img_d, lt = self.val_loader.dataset.untransform(img_o, img_d, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img_d, n_class=n_class)
                    visualizations.append(viz)
        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class, ignore=self.ignore)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter_test_%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        test_loss /= len(self.test_loader)
        test_loss_seg /= len(self.test_loader)
        test_loss_gram /= len(self.test_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('America/New_York')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + [test_loss] + list(metrics[0:-1]) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # logging information for tensorboard
        info = OrderedDict({
            "loss_seg": test_loss_seg,
            "loss_gram": test_loss_gram,
            "loss": test_loss,
            "acc": metrics[0],
            "acc_cls": metrics[1],
            "meanIoU": metrics[2],
            "fwavacc": metrics[3],
            "bestIoU": self.best_mean_iu,
        })
        for i in range(len(self.test_loader.dataset.class_names)):
            if i != self.ignore:
                info['IoU '+self.test_loader.dataset.class_names[i]] = metrics[4][i]
        partial_epoch = self.iteration / len(self.train_loader)
        for tag, value in info.items():
            self.ts_logger.scalar_summary(tag, value, partial_epoch)

        if training:
            self.model.train()

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        val_loss_seg = 0
        val_loss_gram = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data_o, data_d, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data_o, data_d, target = data_o.cuda(), data_d.cuda(), target.cuda()
            data_o, data_d, target = Variable(data_o), Variable(data_d), Variable(target)
            score, score_, gram, gram_ = self.model(data_o, data_d)
            # segmentation loss
            loss_seg = utils.cross_entropy2d(score_, target, size_average=self.size_average)
            # gram loss
            loss_gram = 0
            for g, g_ in zip(gram, gram_):
                loss_gram += F.mse_loss(utils.gram_matrix(g), utils.gram_matrix(g_))
            # loss_gram /= len(gram)
            # overall loss
            loss = loss_seg + loss_gram
            # loss data
            loss_data = float(loss.data[0])
            loss_seg_data = float(loss_seg.data[0])
            loss_gram_data = float(loss_gram.data[0])

            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data_o)
            val_loss_seg += loss_seg_data / len(data_o)
            val_loss_gram += loss_gram_data / len(data_o)

            imgs_o = data_o.data.cpu()
            imgs_d = data_d.data.cpu()
            # predition from the student network -- score_
            lbl_pred = score_.data.max(1)[1].cpu().numpy()[:, :, :].astype(np.int8)
            lbl_true = target.data.cpu().numpy()
            for img_o, img_d, lt, lp in zip(imgs_o, imgs_d, lbl_true, lbl_pred):
                img_o, img_d, lt = self.val_loader.dataset.untransform(img_o, img_d, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img_d, n_class=n_class)
                    visualizations.append(viz)
        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class, ignore=self.ignore)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter_val_%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)
        val_loss_seg /= len(self.val_loader)
        val_loss_gram /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('America/New_York')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + [val_loss] + list(metrics[0:-1]) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        # logging information for tensorboard
        info = OrderedDict({
            "loss_seg": val_loss_seg,
            "loss_gram": val_loss_gram,
            "loss": val_loss,
            "acc": metrics[0],
            "acc_cls": metrics[1],
            "meanIoU": metrics[2],
            "fwavacc": metrics[3],
            "bestIoU": self.best_mean_iu,
        })
        for i in range(len(self.val_loader.dataset.class_names)):
            if i != self.ignore:
                info['IoU '+self.val_loader.dataset.class_names[i]] = metrics[4][i]
        partial_epoch = self.iteration / len(self.train_loader)
        for tag, value in info.items():
            self.v_logger.scalar_summary(tag, value, partial_epoch)

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data_o, data_d, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()
                # self.test()
                # pass

            assert self.model.training

            if self.cuda:
                data_o, data_d, target = data_o.cuda(), data_d.cuda(), target.cuda()
            data_o, data_d, target = Variable(data_o), Variable(data_d), Variable(target)
            self.optim.zero_grad()
            score, score_, gram, gram_ = self.model(data_o, data_d)
            weights = torch.from_numpy(self.train_loader.dataset.class_weights).float().cuda()
            loss_seg = utils.cross_entropy2d(score_, target, weight=weights,
                                             size_average=self.size_average, ignore=self.ignore)
            loss_gram = 0
            for g, g_ in zip(gram, gram_):
                loss_gram += F.mse_loss(utils.gram_matrix(g), utils.gram_matrix(g_))
            # loss_gram /= len(gram)

            loss = loss_seg + loss_gram
            loss /= len(data_o)
            loss_data = float(loss.data[0])
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()
            self.poly_lr_scheduler(iteration)

            metrics = []
            ius = []
            # predition from the student network -- score_
            lbl_pred = score_.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc, iu = utils.label_accuracy_score(lbl_true, lbl_pred, n_class=n_class,
                                                                            ignore=self.ignore)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            ius.append(iu)
            metrics = np.mean(metrics, axis=0)
            ius = np.nanmean(ius, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('America/New_York')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + list(metrics) + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            # logging to tensorboard
            self.best_train_meanIoU = max(self.best_train_meanIoU, metrics[2])
            info = OrderedDict({
                "loss_seg": loss_seg.data[0],
                "loss_gram": loss_gram.data[0],
                "loss": loss.data[0],
                "acc": metrics[0],
                "acc_cls": metrics[1],
                "meanIoU": metrics[2],
                "fwavacc": metrics[3],
                "bestIoU": self.best_train_meanIoU,
            })
            for i in range(len(self.train_loader.dataset.class_names)):
                if i != self.ignore:
                    info['IoU '+self.train_loader.dataset.class_names[i]] = ius[i]
            partialEpoch = self.epoch + float(batch_idx) / len(self.train_loader)
            for tag, value in info.items():
                self.t_logger.scalar_summary(tag, value, partialEpoch)

            if self.iteration >= self.max_iter:
                break

    def poly_lr_scheduler(self, iter, power=0.9):
        lr = self.init_lr * (1 - iter / self.max_iter) ** power
        self.optim.param_groups[0]['lr'] = lr
        self.optim.param_groups[1]['lr'] = 2*lr

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
