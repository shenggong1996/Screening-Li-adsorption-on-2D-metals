#This code is for picking up the most uncertain data points from a set of possible data points to be sampled
import os
import numpy as np
from pymatgen.io.vasp import Poscar
from pymatgen.core.structure import Structure


import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import CIFData
from cgcnn.data import collate_pool
from cgcnn.model import CrystalGraphConvNet


def predict(args, dataset, collate_fn, test_loader):

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task ==
                                'classification' else False)
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    normalizer = Normalizer(torch.zeros(3))

    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    validate(test_loader, model, criterion, normalizer, test=True)


def validate(val_loader, model, criterion, normalizer, test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        if model_args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        with torch.no_grad():
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if model_args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if model_args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       accu=accuracies, prec=precisions, recall=recalls,
                       f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if model_args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    Parameters
    ----------
    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

model_source = 'train_4'  # directory where the models used for prediction from the last batch

n_models = 5 # number of models of the committee

percent = 0.2 # ratio of data points to be sampled

n_batch = 4 # number of current batch

#n_atoms_limit = 20 

#record sampled data
'''
#existed_data = [] 

#f=open('train_%d/sample/id_prop.csv'%(n_batch))
#length=len(f.readlines())
#f.close()
#f=open('train_%d/sample/id_prop.csv'%(n_batch))
#for i in range(length):
#    ID, gap = f.readline().split(',')
#    existed_data.append(ID)
#f.close()
'''

os.system('mkdir batch_%d_prediction_before'%(n_batch))

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
    #parser.add_argument('modelpath', help='path to the trained model.')
#args.modelpath = '%s/model_0.010000_%d.pth.tar'%(model_source)
    #parser.add_argument('cifpath', help='path to the directory of CIF files.')
#    args.cifpath = 'sample_mof/'
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

#args.cuda = not args.disable_cuda and torch.cuda.is_available()
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

args.cifpath = 'candidate_4/'#directory with candidate samples
#args.cifpath = '../hse_sample/'


dataset = CIFData(args.cifpath)
collate_fn = collate_pool
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn, pin_memory = args.cuda)

for i in range(n_models):
    args.modelpath = '%s/model_%d.pth.tar'%(model_source,i)
    if os.path.isfile(args.modelpath):
        print("=> loading model params '{}'".format(args.modelpath))
        model_checkpoint = torch.load(args.modelpath,
                                      map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
        print("=> loaded model params '{}'".format(args.modelpath))
    else:
        print("=> no model params found at '{}'".format(args.modelpath))

    if model_args.task == 'regression':
        best_mae_error = 1e10
    else:
        best_mae_error = 0.
    predict(args, dataset, collate_fn, test_loader)
    os.system('mv test_results.csv batch_%d_prediction_before/test_results_%d.csv'%(n_batch, i)) 

preds = {}

for i in range(n_models):
#    pred = np.genfromtxt('test_results_%d.csv'%(i), delimiter = ',', dtype= None)
#    print (np.shape(pred))
    f=open('batch_%d_prediction_before/test_results_%d.csv'%(n_batch, i))
    length=len(f.readlines())
    f.close()
    f=open('batch_%d_prediction_before/test_results_%d.csv'%(n_batch, i))
    for j in range(length):
        ID,_,pred = f.readline().split(',')
        pred = float(pred)
        if ID not in preds.keys():
            preds[ID] = [pred]
        else:
            preds[ID].append(pred)

uncertainty = {}

IDs = []

for ID in preds.keys():
    uncertainty[ID] = np.std(preds[ID])

sorted_uncertainty = sorted(uncertainty.items(), key=lambda kv: kv[1], reverse=True)

os.system('mkdir 1028_%d_to_be_calculated'%(n_batch))#directory where to be calculated samples are stored

i = 0; n = 0
while n < (int(percent*length)):
    (ID, _) = sorted_uncertainty[i]
#    if ID in existed_data:
#        i+=1
#        continue
    structure = Structure.from_file('candidate_4/%s.cif'%(ID))
#    if not len(structure.species) < n_atoms_limit:
#        i+=1
#        continue
    n+=1; i+=1
#    IDs.append(ID)
    structure.to(filename='1028_%d_to_be_calculated/%s'%(n_batch, ID), fmt='poscar')
    print ('%s, %f, %f'%(ID, np.mean(preds[ID]), np.std(preds[ID])), file=open('1028_%d_predictions.csv'%(n_batch),'a'))

