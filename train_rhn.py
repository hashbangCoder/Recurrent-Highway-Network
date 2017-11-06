import argparse
import time

import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import dataloader
import recurrent_highway_network as models


parser = argparse.ArgumentParser(description='PyTorch Recurrent Highway Network Language Model')

parser.add_argument('--data-path', dest='data_path', type=str, default='data/Text8_char.hdf5',
                    help='Path to preprocessed hdf5 dataset')
parser.add_argument('--input-size', dest='input_size', type=int, default=400, help='size of word embeddings')
parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=1150, help='number of hidden units per layer')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=80, metavar='N', help='batch size')

parser.add_argument('recurrence-length', dest='recurrence_length',
                    type=int, default=3, help='Recurrence Length (L)')
parser.add_argument('--bptt', dest='max_seq_len', type=int, default=70, help='sequence length')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')

parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--grad-clip', dest='grad_clip', type=float, default=0.5, help='gradient clipping, set to -1.0 to disable')
parser.add_argument('--epochs', dest='num_epochs', type=int, default=20, help='upper epoch limit')

# parser.add_argument('--tied', dest='tied_weights', action='store_false',
#                     help='tie the word embedding and softmax weights')
parser.add_argument('--seed', dest='seed', type=int, default=1111, help='random seed')
parser.add_argument('--save', dest='save_path',type=str, help='path to save the final model')
parser.add_argument('--save-best', dest='save_best',type=bool, help='Save the best model', default=True)


opt = parser.parse_args()

assert torch.cuda.is_available(), 'GPU Acceleration not available. Check if GPU is detected and CUDA toolkit is installed'


# Get data
# corpus = data.Corpus(args.data_path)
#
# eval_batch_size = 10
# test_batch_size = 1
# train_data = data.batchify(corpus.train, args.batch_size, args)
# val_data = data.batchify(corpus.valid, eval_batch_size, args)
# test_data = data.batchify(corpus.test, test_batch_size, args)

dl = dataloader.Text8DataLoader(opt.data_path, opt.max_epochs)
model = models.RecurrentHighway(opt.input_size, opt.hidden_size, opt.recurrence_length)
model = model.cuda()
best_perpl = math.inf
optim = torch.optim.Adam(model.parameters(), opt.lr)

def calculate_perplexity(indices, labels):
    perpl = 0
    return perpl


def eval_model(model, save_best=True):
    model.eval()

    eval_batch = dl.get_eval_batch(opt.batch_size, opt.max_seq_length)
    input_batch, label_batch = eval_batch
    input_batch = Variable(torch.from_numpy(input_batch).cuda())

    model_out = F.softmax(model(input_batch))
    _, inds = model_out.max(1)
    perpl = calculate_perplexity(inds, label_batch)
    print('Evaluation perplexity: ', perpl)
    if save_best and eval_perpl < best_perpl:
        with open(opt.save_path + '_{}_{}'.format(dl.epoch, dl.step_counter)) as f:
            torch.save(model, f)

    model.train()
    return perpl

while dl.end_flag is False:
    data_batch = dl.get_batch(opt.batch_size, opt.max_seq_length)
    if data_batch is None:
        print('End of training')
        break

    input_batch, label_batch = data_batch
    input_batch = Variable(torch.from_numpy(input_batch).cuda())
    label_batch = Variable(torch.from_numpy(label_batch).cuda())
    model_out = model(input_batch)
    loss = nn.CrossEntropyLoss(model_out, label_batch)
    loss.backward()

    if opt.grad_clip != -1:
        nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip)
    optim.step()

    if dl.step_counter % opt.eval_freq == 0:
        eval_perpl = eval_model(model, save_best=opt.save_best)
        if eval_perpl < best_perpl:
            best_perpl = eval_perpl

    # TODO : Add plotting of training