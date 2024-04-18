import copy
import time
import torch
import ssl
import random
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import configargparse
from utils import data_loader
from utils.tools import str2bool, AverageMeter, save_model
from models.make_model import TransferNet
import os
from models import rst
import logging

from torch.cuda.amp import GradScaler, autocast
ssl._create_default_https_context = ssl._create_unverified_context
scaler = GradScaler()
def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--datasets', type=str, default='office_home',choices=["office_home","office31","visda",
                                                                               "domain_net","digits","image_clef"])
    parser.add_argument('--use_amp', type=str2bool, default=False)

    # network related
    parser.add_argument('--model_name', type=str, default='RN50',choices=["RN50", "VIT-B", "RN101"])

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)

    # training related
    parser.add_argument('--l_batch_size', type=int, default=32)
    parser.add_argument('--u_batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument("--n_iter_per_epoch", type=int, default=500, help="Used in Iteration-based training")
    parser.add_argument('--rst_threshold', type=float, default=1e-5)
    parser.add_argument('--baseline', default=False, action='store_true')
    parser.add_argument('--pda', default=False, action='store_true')
    parser.add_argument('--rst', default=False, action='store_true')
    parser.add_argument('--clip', default=False, action='store_true')

    # FixMatch
    parser.add_argument('--fixmatch', default=False, action='store_true')
    parser.add_argument('--fixmatch_threshold', type=float, default=0.95)
    parser.add_argument('--fixmatch_factor', type=float, default=0.5)

    # optimizer related
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--multiple_lr_classifier', type=float, default=10)

    # loss related
    parser.add_argument('--lambda1', type=float, default=0.25)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=0.025)
    parser.add_argument('--clf_loss', type=str, default="cross_entropy")

    # learning rate scheduler related
    parser.add_argument('--scheduler', type=str2bool, default=True)

    # linear scheduler
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)

    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    # Use FixMatch
    use_fixmatch = args.fixmatch
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        args, folder_src, args.l_batch_size, infinite_data_loader=True, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        args, folder_tgt, args.u_batch_size, infinite_data_loader=True, train=True, use_fixmatch=use_fixmatch, num_workers=args.num_workers, partial=args.pda)
    target_test_loader, _ = data_loader.load_data(
        args, folder_tgt, args.u_batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers, partial=args.pda)
    return source_loader, target_train_loader, target_test_loader, n_class


def get_model(args):
    model = TransferNet(args).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    return optimizer

def get_lr_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  (args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)))
    return scheduler


def test(model, target_test_loader, args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    first_test = True
    desc = "Clip Testing..." if args.clip else "Testing..."
    with torch.no_grad():
        for data, target in tqdm(iterable=target_test_loader,desc=desc):
            data, target = data.to(args.device), target.to(args.device)
            if args.clip:
                s_output = model.clip_predict(data)
            else:
                s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            if first_test:
                all_pred = pred
                all_label = target
                first_test = False
            else:
                all_pred = torch.cat((all_pred, pred), 0)
                all_label = torch.cat((all_label, target), 0)

    if args.datasets == "visda":
        acc = metrics.balanced_accuracy_score(all_label.cpu().numpy(),
                                                          torch.squeeze(all_pred).float().cpu().numpy()) *100
        cm = metrics.confusion_matrix(all_label.cpu().numpy(),
                                              torch.squeeze(all_pred).float().cpu().numpy())
        per_classes_acc = list(((cm.diagonal() / cm.sum(1))*100).round(4))
        per_classes_acc = list(map(str, per_classes_acc))
        per_classes_acc = ', '.join(per_classes_acc)
        if args.clip:
            print('CLIP: test_loss {:4f}, test_acc: {:.4f} \nper_class_acc: {}'.format(test_loss.avg, acc, per_classes_acc))
        else:
            return acc, per_classes_acc, test_loss.avg
    else:
        acc = torch.sum(torch.squeeze(all_pred).float() == all_label) / float(all_label.size()[0]) * 100
        if args.clip:
            print('CLIP: test_loss {:4f}, test_acc: {:.4f}'.format(test_loss.avg, acc))
        else:
            return acc, test_loss.avg

def obtain_label(model,loader,e,args):
    # For partial-set domain adaptation on the office-home benchmark
    model.eval()
    class_set = []
    if e==1:
        return [i for i in range(65)]
    number_threshold = 14
    classes_num = [0 for _ in range(65)]
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(args.device)
            s_output = model.predict(data)
            preds = torch.max(s_output, 1)[1]
            for pred in preds:
                classes_num[pred] += 1
    for c,n in enumerate(classes_num):
        if n >= number_threshold:
            class_set.append(c)
    return class_set

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args):
    logging.basicConfig(filename=os.path.join(args.log_dir,'training.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    n_batch = args.n_iter_per_epoch
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    for e in range(1, args.n_epoch+1):
        if args.pda:
            assert args.datasets=="office_home"
            label_set = obtain_label(model, target_test_loader, e, args)
        else:
            label_set = None

        model.train()

        train_loss_clf = AverageMeter()
        train_loss_transfer = AverageMeter()
        train_loss_total = AverageMeter()

        for _ in tqdm(iterable=range(n_batch),desc=f"Train:[{e}/{args.n_epoch}]"):
            optimizer.zero_grad()
            data_source, label_source = next(iter_source) # .next()
            data_target, _ = next(iter_target) # .next()
            data_source, label_source = data_source.to(args.device), label_source.to(args.device)
            data_target_strong = None
            if args.fixmatch:
                data_target, data_target_strong = data_target[0], data_target[1]
                data_target, data_target_strong = data_target.to(args.device), data_target_strong.to(args.device)
            else:
                data_target = data_target.to(args.device)
            if args.use_amp:
                # mixture precision
                with autocast():
                    clf_loss, transfer_loss = model(data_source, data_target, label_source, data_target_strong)
                    loss = clf_loss + transfer_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # fully precision
                clf_loss, transfer_loss = model(data_source, data_target, label_source, data_target_strong, label_set)
                loss = clf_loss + transfer_loss
                loss.backward()
                optimizer.step()

            if args.rst:
                rst.training(model,args)

            # learning rate scheduler update
            scheduler.step()

            # training loss update
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())

        # Test
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        if args.datasets == "visda":
            test_acc, test_per_class_acc, test_loss = test(model, target_test_loader, args)
            info += ', test_loss {:4f}, test_acc: {:.4f} \nper_class_acc: {}'.format(test_loss, test_acc, test_per_class_acc)
        else:
            test_acc, test_loss = test(model, target_test_loader, args)
            info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)

        if args.rst:
            dsp = rst.dsp_calculation(model)
            info += ', dsp: {:.4f}'.format(dsp)

        if best_acc < test_acc:
            best_acc = test_acc
            save_model(model,args)

        logging.info(info)
        tqdm.write(info)
        time.sleep(1)

    tqdm.write('Transfer result: {:.4f}'.format(best_acc))


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, num_class = load_data(args)
    setattr(args, "num_class", num_class)
    setattr(args, "max_iter", 10000)
    log_dir = f'log/{args.model_name}/{args.datasets}/{args.src_domain}2{args.tgt_domain}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setattr(args, "log_dir", log_dir)
    print(args)
    model = get_model(args)
    print(model)
    optimizer = get_optimizer(model, args)

    if args.scheduler:
        scheduler = get_lr_scheduler(optimizer,args)
    else:
        scheduler = None
    print(f"Base Network: {args.model_name}")
    print(f"Source Domain: {args.src_domain}")
    print(f"Target Domain: {args.tgt_domain}")
    print(f"FixMatch: {args.fixmatch}")
    print(f"Residual Sparse Training: {args.rst}")
    if args.rst:
        print(f"Residual Sparse Training Threshold: {args.rst_threshold}")
    if args.clip:
        test(model, target_test_loader, args)
    else:
        train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    

if __name__ == "__main__":
    main()
