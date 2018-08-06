#!/usr/bin/env python
import os
import torch
import utils
import models
import torchfcn
import datasets
import argparse
import os.path as osp
from network_trainer import Trainer


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    )
}


def main():
    # 0. input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, help='GPU device to use', default=1)
    parser.add_argument('-b', '--batch', type=int, help='batch size', default=1)
    parser.add_argument('-d', '--dataset', help='VOC, CamVid, SUNRGBD', default='SUNRGBD')
    parser.add_argument('-dr', '--datasetroot', help='dataset root pth', default='/home/dg/Dropbox/Datasets')
    parser.add_argument('-de', '--degradation', help='o, bg, bm, h, ns, nsp', default='o')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
    parser.add_argument('-r', '--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    batch = args.batch
    dataset = args.dataset
    dataset_root = args.datasetroot
    degradation = args.degradation
    cfg = configurations[args.config]
    out = utils.get_log_dir('fcn8s-atonce', args.config, cfg)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    root = osp.expanduser(osp.join(dataset_root, dataset))
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    if dataset == 'VOC':
        train_data = datasets.VOCSeg(root, split='train', dataset=degradation, transform=True)
        val_data = datasets.VOCSeg(root, split='val', dataset=degradation, transform=True)
        test_data = datasets.VOCSeg(root, split='test', dataset=degradation, transform=True)
    elif dataset == "CamVid":
        train_data = datasets.CamVidSeg(root, split='train', dataset=degradation, transform=True)
        val_data = datasets.CamVidSeg(root, split='val', dataset=degradation, transform=True)
        test_data = datasets.CamVidSeg(root, split='test', dataset=degradation, transform=True)
    else:
        train_data = datasets.SUNSeg(root, split='train', dataset=degradation, transform=True)
        val_data = datasets.SUNSeg(root, split='val', dataset=degradation, transform=True)
        test_data = datasets.SUNSeg(root, split='test', dataset=degradation, transform=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=False, **kwargs)

    # 2. model
    model = models.FCN8sAtOnce(n_class=train_data.n_classes)
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)

    # 3. optimizer
    optim = torch.optim.SGD(
        [
            {'params': utils.get_parameters(model, bias=False)},
            {'params': utils.get_parameters(model, bias=True),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    # 4. trainer
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=test_loader,
        test_loader=test_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
