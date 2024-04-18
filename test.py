import torch
import ssl
import random
import numpy as np
from tqdm import tqdm
import configargparse
from utils import data_loader
from utils.tools import str2bool, AverageMeter, load_checkpoint
from models.make_model import TransferNet
import os
from sklearn import metrics

ssl._create_default_https_context = ssl._create_unverified_context
import torch.sparse

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--datasets', type=str, default='office_home', choices=["office_home", "office31", "visda",
                                                                                "domain_net", "digits", "image_clef"])

    # network related
    parser.add_argument('--model_name', type=str, default='RN50', choices=["RN50", "VIT-B", "RN101"])

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)

    # testing related
    parser.add_argument('--rst', default=False, action='store_true')
    parser.add_argument('--clip', default=False, action='store_true')

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
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    target_test_loader, n_class = data_loader.load_data(
        args, folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return target_test_loader, n_class


def get_model(args):
    model = TransferNet(args,train=False)
    return model


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
        print('test_loss {:4f}, test_acc: {:.4f} \nper_class_acc: {}'.format(test_loss.avg, acc, per_classes_acc))
    else:
        acc = torch.sum(torch.squeeze(all_pred).float() == all_label) / float(all_label.size()[0]) * 100
        print('test_loss {:4f}, test_acc: {:.4f}'.format(test_loss.avg, acc))


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    set_random_seed(args.seed)
    target_test_loader, num_class = load_data(args)
    setattr(args, "num_class", num_class)
    log_dir = f'log/{args.model_name}/{args.datasets}/{args.src_domain}2{args.tgt_domain}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setattr(args, "log_dir", log_dir)
    print(args)
    model = get_model(args)
    print(f"Base Network: {args.model_name}")
    print(f"Source Domain: {args.src_domain}")
    print(f"Target Domain: {args.tgt_domain}")
    if args.rst:
        print(f"Residual Sparse Training: {args.rst}")
    if args.clip:
        print(f"CLIP Inference: {args.clip}")


    if not args.clip:
        model = load_checkpoint(model,args)

    test(model, target_test_loader, args)


if __name__ == "__main__":
    main()