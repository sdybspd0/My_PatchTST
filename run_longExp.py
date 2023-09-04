import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
parser.add_argument('--nei_layers', type=int,  default=1,help='周期内的层数')
parser.add_argument('--jian_layers', type=int,  default=1,help='周期间的层数')
parser.add_argument('--chong_len', type=int,  default=8,help='每个patch的重叠大小')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--e_layers', type=int, default=2, help='num of patchTST layers')
parser.add_argument('--local_bias', type=float, default=0.5, help='pred = pred + local_bias*local_output')
parser.add_argument('--global_bias', type=float, default=0.5, help='pred = pred + global_bias*global_output ')
parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')
parser.add_argument('--fc_dropout', type=float, default=0.3, help='fully connected dropout')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
# random seed
parser.add_argument('--random_seed', type=int, default=2023, help='random seed')

# basic config
parser.add_argument('--model', type=str, default='GCformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--context_len', type=int, default=96, help='input sequence length for local_model')
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length for global_model')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# PatchTST

parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')

parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--local_revin', type=int, default=0, help='RevIN for local_model(PatchTST)')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')


# GCFormer 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--enc_raw', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=16, help='num of heads')

parser.add_argument('--global_layers', type=int, default=1, help='num of global layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)

parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--h_token', type=int, default=512, help='dimension of model')
parser.add_argument('--h_channel', type=int, default=128, help='dimension of model')

# optimization
parser.add_argument('--perturb_ratio', type=float, default=0.0, help='noise ratio')
parser.add_argument('--global_model', type=str, default='Gconv', help='Gconv FNO Film')
parser.add_argument('--norm_type', type=str, default='revin', help='revin  seq_last')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='dropout')

parser.add_argument('--atten_bias', type=float, default=0.5)
parser.add_argument('--TC_bias', type=float, default=1, help='1:attention over channel  0:attention over token ')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_sl{}_cl{}_pl{}_nchannel{}_ntoken{}_nhead{}_d{}_df{}el{}_dl{}_attenBias{}_TCbias{}_dp{}_Lbias{}_Gbias{}_{}_noise{}_decay{}_lr{}_decompose{}_individual{}'.format(
            args.model,
            args.global_model,
            args.data,
            args.seq_len,
            args.context_len,
            args.pred_len,
            args.h_channel,
            args.h_token,
            args.n_heads,
            args.d_model,
            args.d_ff,
            args.e_layers,
            args.d_layers,
            args.atten_bias,
            args.TC_bias,
            args.fc_dropout,
            args.local_bias,
            args.global_bias,
            args.norm_type,
            args.perturb_ratio,
            args.weight_decay,
            args.learning_rate,
            args.decomposition,
            args.individual       
)

    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    print('无layernorm')
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
