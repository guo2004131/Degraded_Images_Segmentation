#!/usr/bin/env python
import os
import torch
import utils
import models
import torchfcn
import datasets
import argparse
import os.path as osp
from network_trainer_densegram import TrainerDG


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=200000,
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
    parser.add_argument('-d', '--dataset', help='VOC, CamVid, SUNRGBD', default='CamVid')
    parser.add_argument('-dp', '--datasetpth', help='dataset root pth', default='/home/dg/Dropbox/Datasets')
    parser.add_argument('-de', '--degradation', help='o, bg, bm, h, ns, nsp', default='h1.5')
    parser.add_argument('-ft', '--freezeteacher', help='free teacher network parameters', default=0)
    parser.add_argument('-fs', '--freezestudent', help='free student network parameters', default=1)
    parser.add_argument('-mp', '--modelpth', help='root pth to models', default='/home/dg/data/models/DegradedImage')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
    parser.add_argument('-r', '--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    batch = args.batch
    dataset = args.dataset
    dataset_pth = args.datasetpth
    degradation = args.degradation
    freezeteacher = args.freezeteacher
    freezestudent = args.freezestudent
    modelpth = args.modelpth
    cfg = configurations[args.config]
    out = utils.get_log_dir('fcn8s-atonce', args.config, cfg)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    root = osp.expanduser(osp.join(dataset_pth, dataset))
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    if dataset == 'VOC':
        train_data = datasets.VOCSegDG(root, split='train', dataset=degradation, transform=True)
        val_data = datasets.VOCSegDG(root, split='val', dataset=degradation, transform=True)
        test_data = datasets.VOCSegDG(root, split='test', dataset=degradation, transform=True)
        model_filename = osp.join(modelpth, 'VOC_fcn8satonce.pth.tar')
    elif dataset == "CamVid":
        train_data = datasets.CamVidSegDG(root, split='train', dataset=degradation, transform=True)
        val_data = datasets.CamVidSegDG(root, split='val', dataset=degradation, transform=True)
        test_data = datasets.CamVidSegDG(root, split='test', dataset=degradation, transform=True)
        model_filename = osp.join(modelpth, 'CamVid_fcn8satonce.pth.tar')
    else:
        train_data = datasets.SUNSegDG(root, split='train', dataset=degradation, transform=True)
        val_data = datasets.SUNSegDG(root, split='val', dataset=degradation, transform=True)
        test_data = datasets.SUNSegDG(root, split='test', dataset=degradation, transform=True)
        model_filename = osp.join(modelpth, 'SUNRGBD_fcn8satonce.pth.tar')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=False, **kwargs)

    # 2. model
    model = models.FCN8sAtOnceDenseGram(n_class=train_data.n_classes)
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        # TODO:load fcn8s model
        fcn8s = torch.torch.load(model_filename)
        fcn8s = fcn8s['model_state_dict']
        model.copy_params_from_vgg16_fcn8s(vgg16, fcn8s)
        # model.copy_params_from_fcn8s(fcn8s)
        # model.copy_params_from_vgg16(vgg16)
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
    trainer = TrainerDG(
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
    model.freeze_params(freezeteacher, freezestudent)
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
