__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.fft
import torch.nn.functional as F
import numpy as np
import ast
import numpy as np
import matplotlib.pyplot as plt
#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from layers.Embed import *
from layers.Embed import DataEmbedding

def FFT_for_Period(x, k=2): #(896,336,32)(896,336,1)
    xf = torch.fft.rfft(x, dim=1) #(896,169,32) (896,336,1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1) #(169)
    _, top_list = torch.topk(frequency_list, k) #1
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

# class PeriodBlock(nn.Module):  # i means channel-independent
#     def __init__(self, period_list: ast.literal_eval, top_k, revin, affine, subtract_last, c_in, jian_layers: int, nei_layers: int, context_window: int, patch_num, patch_len, max_seq_len=1024,
#                  n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
#                  d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
#                  key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
#                  pe='zeros', learn_pe=True, verbose=False):
#         super().__init__()
#
#         # RevIn
#         self.revin = revin
#         if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
#         self.top_k = top_k
#         self.context_window = context_window
#         self.period_list = period_list
#
#     def forward(self, z, z_mark_enc) -> Tensor: #(128,7,336) (128,336,4)
#         # norm
#         if self.revin:
#             z = z.permute(0, 2, 1)  # (128,336,7)
#             z = self.revin_layer(z, 'norm')  # (128,336,7)
#             z = z.permute(0, 2, 1)  # (128,7,336)
#
#         # 自己写的
#         n_vars = z.shape[1]  # 7
#         # z = z.permute(0, 2, 1)  # batch c len (128,7,336)
#         z = torch.reshape(z, (z.shape[0] * z.shape[1], z.shape[2]))  # batch*c len #(896,336)
#         z = z.unsqueeze(-1)  # (224,96,1) (896,336,1)
#         z_mark_enc = z_mark_enc.repeat_interleave(n_vars, dim=0)  # 假期编码重复7遍 (896,336,4)
#
#         # embedding
#         # z = self.enc_embedding(z, z_mark_enc)  # [B,T,C] (896,336,32)
#
#         B, T, N = z.size()  # B 是 batch size，T 是时间序列长度，N 是特征的维度
#         period_l, period_weight = FFT_for_Period(z, self.top_k)
#         self.period_list = period_l
#
#         for i in range(self.top_k):
#             period = self.period_list[i]
#             if self.context_window % period != 0:
#                 length = (( self.context_window // period ) + 1) * period
#                 padding = torch.zeros([z.shape[0], (length - self.context_window), z.shape[2]]).to(z.device)
#                 out = torch.cat([z, padding], dim=1)
#             else:
#                 length = self.context_window
#                 out = z
#             z = out.reshape(B, length // period, period,N).permute(0, 3, 1, 2).contiguous() #(896,1,3,112)
#
#         return z, self.period_list


class MLP_backbone(nn.Module):
    def __init__(self,chong_len: int, jian_layers: int, nei_layers: int, embed: str, freq: str, vars: int, d_embed: int, top_k: int,c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, **kwargs):

        super().__init__()

        self.nei_layers = nei_layers
        self.jian_layers = jian_layers
        # self.period_list = period_list
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0,chong_len))
            self.patch_num += 1

        self.context_window = context_window
        # self.top_k = top_k


        # Backbone
        self.backbone = 1
        # self.backbone = MLPEncoder(period_list, top_k,  revin, affine, subtract_last, c_in, jian_layers = jian_layers, nei_layers = nei_layers, context_window = context_window, patch_num=self.patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
        #                             n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
        #                             attn_dropout=attn_dropout, dropout=dropout, act=act,
        #                             key_padding_mask=key_padding_mask, padding_var=padding_var,
        #                             attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
        #                             store_attn=store_attn,
        #                             pe=pe, learn_pe=learn_pe, verbose=verbose)


        # self.period_block = PeriodBlock(period_list, top_k, revin, affine, subtract_last, c_in, jian_layers=jian_layers, nei_layers=nei_layers, context_window=context_window,
        #                            patch_num=self.patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
        #                            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
        #                            attn_dropout=attn_dropout, dropout=dropout, act=act,
        #                            key_padding_mask=key_padding_mask, padding_var=padding_var,
        #                            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
        #                            store_attn=store_attn,
        #                            pe=pe, learn_pe=learn_pe, verbose=verbose)

        # Head
        self.head_nf = patch_len * self.patch_num   #已改
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        self.vars = 1
        self.enc_embedding = DataEmbedding(vars, d_embed, embed, freq, dropout)
        self.top_k = top_k
        self.context_window = context_window

        #
        self.affine = affine
        self.subtract_last = subtract_last
        self.c_in = c_in
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.act = act
        self.key_padding_mask = key_padding_mask
        self.padding_var = padding_var
        self.attn_mask = attn_mask
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.pe = pe
        self.learn_pe = learn_pe
        self.verbose = verbose
        self.max_seq_len = max_seq_len
        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def get_period(self, period_list):
        device = torch.device("cuda")
        self.backbone = MLPEncoder(period_list, self.n_vars, self.top_k, self.revin, self.affine, self.subtract_last, self.c_in, jian_layers=self.jian_layers,
                                   nei_layers=self.nei_layers, context_window=self.context_window, patch_num=self.patch_num,
                                   patch_len=self.patch_len, max_seq_len=self.max_seq_len,
                                   n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads, d_k=self.d_k, d_v=self.d_v, d_ff=self.d_ff,
                                   attn_dropout=self.attn_dropout, dropout=self.dropout, act=self.act,
                                   key_padding_mask=self.key_padding_mask, padding_var=self.padding_var,
                                   attn_mask=self.attn_mask, res_attention=self.res_attention, pre_norm=self.pre_norm,
                                   store_attn=self.store_attn,
                                   pe=self.pe, learn_pe=self.learn_pe, verbose=self.verbose).to(device)

    def forward(self, z, z_mark_enc):  # (128,7,336)      # z: [bs x nvars x seq_len]
        # z,  period_list = self.period_block(z, z_mark_enc) #(896,1,3,112)
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)  # (128,336,7)
            z = self.revin_layer(z, 'norm')  # (128,336,7)
            z = z.permute(0, 2, 1)  # (128,7,336)

        # 自己写的
        n_vars = z.shape[1]  # 7
        # z = z.permute(0, 2, 1)  # batch c len (128,7,336)
        z = torch.reshape(z, (z.shape[0] * z.shape[1], z.shape[2]))  # batch*c len #(896,336)
        z = z.unsqueeze(-1)  # (224,96,1) (896,336,1)
        z_mark_enc = z_mark_enc.repeat_interleave(n_vars, dim=0)  # 假期编码重复7遍 (896,336,4)

        # embedding
        # z = self.enc_embedding(z, z_mark_enc)  # [B,T,C] (896,336,32)

        B, T, N = z.size()  # B 是 batch size，T 是时间序列长度，N 是特征的维度
        period_list, period_weight = FFT_for_Period(z, self.top_k)


        for i in range(self.top_k):
            period = period_list[i]
            if self.context_window % period != 0:
                length = (( self.context_window // period ) + 1) * period
                padding = torch.zeros([z.shape[0], (length - self.context_window), z.shape[2]]).to(z.device)
                out = torch.cat([z, padding], dim=1)
            else:
                length = self.context_window
                out = z
            z = out.reshape(B, length // period, period,N).permute(0, 3, 1, 2).contiguous() #(896,1,3,112)
            # z = torch.reshape(z, (z.shape[0] * z.shape[1], z.shape[2], z.shape[3])) #(896,3,112)

        m = z[1][0].cpu()
        plt.imshow(m, cmap="viridis")
        plt.colorbar()
        plt.show()
        plt.savefig("kk.jpg")
        # 获取矩阵的形状
        # x, y, z = np.indices(m.shape)
        # # 将三维矩阵展开为一维数组
        # x = x.flatten()
        # y = y.flatten()
        # z = z.flatten()
        # # 将矩阵中的值展开为一维数组
        # values = m.flatten()
        # # 创建一个三维图形对象
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # # 绘制散点图
        # ax.scatter(x, y, z, c=values, cmap='viridis')
        # # 设置坐标轴标签
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # # 显示图形
        # plt.show()
        # plt.savefig("k.jpg")

        # # do patching
        # if self.padding_patch == 'end':
        #     z = self.padding_patch_layer(z)  # (128,7,344) (128,7,104)
        # # #自己写的
        # # if self.padding_patch == 'end':
        # #     row_means = torch.mean(z, dim=2, keepdim=True)
        # #     z= torch.cat([z, row_means.repeat(1, 1, self.stride)], dim=2)
        #
        #
        # # 从一个分批输入的张量中提取滑动的局部块
        # z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (128,7,42,16)  # z: [bs x nvars x patch_num x patch_len]
        # z = z.permute(0, 1, 3, 2)  # (128,7,16,42) (128,7,16,12) (128,7,24,6) (128,7,24,21)       # z: [bs x nvars x patch_len x patch_num]

        # model
        self.get_period(period_list)
        z = self.backbone(z)  # (128,7,16,42) (128,7,3,112)                                                      # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # (128,7,96)                                                              # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)  # (128,96,7)
            z = self.revin_layer(z, 'denorm')  # (128,96,7)
            z = z.permute(0, 2, 1)  # (128,7,96)
        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),nn.Conv1d(head_nf, vars, 1))


class MLPEncoder(nn.Module):  # i means channel-independent
    def __init__(self,  period_list, n_vars, top_k, revin, affine, subtract_last,  c_in,   jian_layers: int, nei_layers: int, context_window: int, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False):
        super().__init__()

        # self.the_period = PeriodBlock(top_k, revin, affine, subtract_last, c_in, jian_layers=jian_layers, nei_layers=nei_layers, context_window=context_window,
        #                            patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
        #                            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
        #                            attn_dropout=attn_dropout, dropout=dropout, act=act,
        #                            key_padding_mask=key_padding_mask, padding_var=padding_var,
        #                            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
        #                            store_attn=store_attn,
        #                            pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        # _, self.period_list = self.the_period.forward()
        self.period_list = period_list
        self.top_k = top_k
        self.context_window = context_window
        # self.max_seq_len = max_seq_len
        self.n_vars = n_vars

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.seq_len = q_len

        self.nei_layers = nei_layers
        self.jian_layers = jian_layers

        for i in range(self.top_k):
            period = self.period_list[i]
            self.mlp_blocks = nn.ModuleList([MixerBlock(self.context_window // period, period, d_model, d_ff, jian_layers, nei_layers, dropout, n_vars) for _ in range(n_layers)])

        # self.mlp_blocks = nn.ModuleList([
        #     MixerBlock(self.patch_num, self.patch_len, d_model, d_ff, jian_layers, nei_layers, dropout) for _ in range(n_layers)])

    def forward(self, x) -> Tensor:  # (128,7,16,42) (896,1,3,112)            # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # (128,7,42,16)  (128,7,6,24)                                          # x: [bs x nvars x patch_num x patch_len]

        z = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  #(896,1,84,4) (896,112,3) （896，42，16）(896,6,24)   # u: [bs * nvars x patch_num x d_model]

        for block in self.mlp_blocks:
            z = block(z)   # （896，42，16） (896,21,24) (896,84,4)

        z = z.transpose(1,2) # （896，16,42） (896,4,84)
        z = torch.reshape(z, ( -1, self.n_vars, z.shape[-2], z.shape[-1]))  # (128,7,16,42)    (896,1,4,84)          # z: [bs x nvars x patch_num x d_model]

        return z #(128, 7, 16, 42) (128,7.3,112)

class MixerBlock(nn.Module):
    def __init__(self, patch_num, patch_len, d_model, d_ff, jian_layers, nei_layers, dropout, n_vars):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(patch_len)
        self.nei_mixings = nn.ModuleList([FactorizedNeiMixing(patch_len, d_model, dropout) for _ in range(nei_layers)])
        # self.layer_norm2 = nn.LayerNorm(patch_num)
        self.jian_mixings = nn.ModuleList([FactorizedJianMixing(patch_num, d_ff, dropout) for _ in range(jian_layers)])

    def forward(self, src:Tensor): # （896，84，4）（896，42，16）(896,112,3)

        y = src # （896，84，4） （896，42，16）(896,21,24)
        y = y.transpose(1, 2) #(896,4,84)
        y = self.layer_norm1(y)
        for nei in self.nei_mixings:
            res1 = y # (896,4,84)（896，42，16）(896,12,16)
            y = nei(y) # （896，42，16）(896,21,24)
            y = res1 + y # （896，42，16）

        y = y.transpose(1, 2)  # （896，16,42）(896,24,21)

        for jian in self.jian_mixings:
            res2 = y  # （896，16,42） (896,84,4)
            # y = self.layer_norm2(y)
            y = jian(y)  # （896，16,42）
            y = res2 + y  # （896，16,42） (896,84,4)
        # y = y.transpose(1, 2) #  (896,21,24)

        return y # (896,84,4)


class FactorizedNeiMixing(nn.Module):
    def __init__(self,patch_len, d_model, dropout) :
        super().__init__()

        # assert input_dim > factorized_dim
        # device = torch.device("cuda")
        # self.nei_mixing = NeiBlock(patch_len, d_model, dropout).to(device)
        self.nei_mixing = NeiBlock(patch_len, d_model, dropout)

    def forward(self, x): #(896,42,16)

        return self.nei_mixing(x)

class FactorizedJianMixing(nn.Module):
    def __init__(self, patch_num, d_ff, dropout) :
        super().__init__()

        # assert input_dim > factorized_dim
        self.jian_mixing = JianBlock(patch_num, d_ff, dropout)

    def forward(self, x): #(896,16,42)

        return self.jian_mixing(x)


class NeiBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):  #(896,4,84) (32,7,42)
        # [B, L, D] or [B, D, L]
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        # return self.fc2(self.gelu(self.fc1(x)))
        return x

class JianBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, input_dim)
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):  # (32,7,42)
        # [B, L, D] or [B, D, L]
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        # x = self.dropout2(x)

        return x
        # return self.fc2(self.gelu(self.fc1(x)))


# class MixerBlock(nn.Module):
#     def __init__(self, patch_num, patch_len, d_model, d_ff, jian_layers, nei_layers, dropout):
#         super().__init__()
#         self.layer_norm1 = nn.LayerNorm(patch_len)
#         self.nei_mixings = nn.ModuleList([FactorizedNeiMixing(patch_len, d_model, dropout) for _ in range(nei_layers)])
#         # self.layer_norm2 = nn.LayerNorm(patch_num)
#         self.jian_mixings = nn.ModuleList([FactorizedJianMixing(patch_num, d_ff, dropout) for _ in range(jian_layers)])
#
#     def forward(self, src:Tensor): # （896，42，16）(896,112,3)
#
#         y = src # （896，42，16）(896,21,24)
#         y = self.layer_norm1(y)
#         for nei in self.nei_mixings:
#             res1 = y # （896，42，16）(896,12,16)
#             y = nei(y) # （896，42，16）(896,21,24)
#             y = res1 + y # （896，42，16）
#
#         y = y.transpose(1, 2)  # （896，16,42）(896,24,21)
#
#         for jian in self.jian_mixings:
#             res2 = y  # （896，16,42）
#             # y = self.layer_norm2(y)
#             y = jian(y)  # （896，16,42）
#             y = res2 + y  # （896，16,42）
#         y = y.transpose(1, 2) #  (896,21,24)
#
#         return y
#
#
# class FactorizedNeiMixing(nn.Module):
#     def __init__(self,patch_len, d_model, dropout) :
#         super().__init__()
#
#         # assert input_dim > factorized_dim
#         self.nei_mixing = NeiBlock(patch_len, d_model, dropout)
#
#     def forward(self, x): #(896,42,16)
#
#         return self.nei_mixing(x)
#
# class FactorizedJianMixing(nn.Module):
#     def __init__(self, patch_num, d_ff, dropout) :
#         super().__init__()
#
#         # assert input_dim > factorized_dim
#         self.jian_mixing = JianBlock(patch_num, d_ff, dropout)
#
#     def forward(self, x): #(896,16,42)
#
#         return self.jian_mixing(x)
#
#
# class NeiBlock(nn.Module):
#     def __init__(self, input_dim, mlp_dim, dropout):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, mlp_dim)
#         self.gelu = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(mlp_dim, input_dim)
#         self.dropout2 = nn.Dropout(dropout)
#
#     def forward(self, x):  # (32,7,42)
#         # [B, L, D] or [B, D, L]
#         x = self.fc1(x)
#         x = self.gelu(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.dropout2(x)
#         # return self.fc2(self.gelu(self.fc1(x)))
#         return x
#
# class JianBlock(nn.Module):
#     def __init__(self, input_dim, mlp_dim, dropout):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, mlp_dim)
#         self.gelu = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(mlp_dim, input_dim)
#         # self.dropout2 = nn.Dropout(dropout)
#
#     def forward(self, x):  # (32,7,42)
#         # [B, L, D] or [B, D, L]
#         x = self.fc1(x)
#         x = self.gelu(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         # x = self.dropout2(x)
#
#         return x
#         # return self.fc2(self.gelu(self.fc1(x)))

class Flatten_Head(nn.Module):
    def __init__(self, individual, weidu, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = weidu

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = 1
            self.dropout = nn.Dropout(head_dropout)

        self.target_window = target_window

    def get_linear(self, patch_num, patch_len, target_window):
        device = torch.device("cuda")
        nf = patch_len * patch_num
        self.linear = nn.Linear(nf, target_window).to(device)


    def forward(self, x):        #(128,7,128,42)                         # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else: #(896,1,3,112)
            B, weidu, patch_len, patch_num = x.size()
            x = self.flatten(x) #(128,7,5376)

            self.get_linear(patch_len, patch_num, self.target_window)
            x = self.linear(x) #(128,7,96)
            x = self.dropout(x) #(128,7,96)
        return x

# #####################################################################################
# # Cell
# class PatchTST_backbone(nn.Module):
#     def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
#                  n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
#                  d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
#                  padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
#                  pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
#                  pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
#                  verbose:bool=False, **kwargs):
#
#         super().__init__()
#
#         # RevIn
#         self.revin = revin
#         if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
#
#         # Patching
#         self.patch_len = patch_len
#         self.stride = stride
#         self.padding_patch = padding_patch
#         patch_num = int((context_window - patch_len)/stride + 1)
#         if padding_patch == 'end': # can be modified to general case
#             self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
#             patch_num += 1
#
#         # Backbone
#         self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
#                                 n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
#                                 attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
#                                 attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
#                                 pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
#
#         # Head
#         self.head_nf = d_model * patch_num
#         self.n_vars = c_in
#         self.pretrain_head = pretrain_head
#         self.head_type = head_type
#         self.individual = individual
#
#         if self.pretrain_head:
#             self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
#         elif head_type == 'flatten':
#             self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
#
#
#     def forward(self, z):     # (128,7,336)                                              # z: [bs x nvars x seq_len]
#         # norm
#         if self.revin:
#             z = z.permute(0,2,1) # (128,336,7)
#             z = self.revin_layer(z, 'norm')  # (128,336,7)
#             z = z.permute(0,2,1) # (128,7,336)
#
#         # do patching
#         if self.padding_patch == 'end':
#             z = self.padding_patch_layer(z) #(128,7,344)
#         # 从一个分批输入的张量中提取滑动的局部块
#         z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  #(128,7,42,16)  # z: [bs x nvars x patch_num x patch_len]
#         z = z.permute(0,1,3,2)  #(128,7,16,42)                                                            # z: [bs x nvars x patch_len x patch_num]
#
#         # model
#         z = self.backbone(z)          #(128,7,128,42)                                                       # z: [bs x nvars x d_model x patch_num]
#         z = self.head(z)       #(128,7,96)                                                              # z: [bs x nvars x target_window]
#
#         # denorm
#         if self.revin:
#             z = z.permute(0,2,1) #(128,96,7)
#             z = self.revin_layer(z, 'denorm') #(128,96,7)
#             z = z.permute(0,2,1) #(128,7,96)
#         return z
#
#     def create_pretrain_head(self, head_nf, vars, dropout):
#         return nn.Sequential(nn.Dropout(dropout),
#                     nn.Conv1d(head_nf, vars, 1)
#                     )
#
#

#
#
#
# class TSTiEncoder(nn.Module):  #i means channel-independent
#     def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
#                  n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
#                  d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
#                  key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
#                  pe='zeros', learn_pe=True, verbose=False, **kwargs):
#
#
#         super().__init__()
#
#         self.patch_num = patch_num
#         self.patch_len = patch_len
#
#         # Input encoding
#         q_len = patch_num
#         self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
#         self.seq_len = q_len
#
#         # Positional encoding
#         self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
#
#         # Residual dropout
#         self.dropout = nn.Dropout(dropout)
#
#         # Encoder
#         self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
#                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
#
#
#     def forward(self, x) -> Tensor:       #(128,7,16,42)             # x: [bs x nvars x patch_len x patch_num]
#
#         n_vars = x.shape[1]
#         # Input encoding
#         x = x.permute(0,1,3,2)        #(128,7,42,16)                                            # x: [bs x nvars x patch_num x patch_len]
#         x = self.W_P(x)               #(128,7,42,128)                                           # x: [bs x nvars x patch_num x d_model]
#
#         u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))  #（896，42，128）    # u: [bs * nvars x patch_num x d_model]
#         u = self.dropout(u + self.W_pos)           #(896,42,128)                            # u: [bs * nvars x patch_num x d_model]
#
#         # Encoder
#         z = self.encoder(u)       #(896,42,128)                                               # z: [bs * nvars x patch_num x d_model]
#         z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))  #(128,7,42,128)              # z: [bs x nvars x patch_num x d_model]
#         z = z.permute(0,1,3,2)     #(128,7,128,42)                                               # z: [bs x nvars x d_model x patch_num]
#
#         return z
#
#
#
# # Cell
# class TSTEncoder(nn.Module):
#     def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
#                         norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
#                         res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
#         super().__init__()
#
#         self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
#                                                       attn_dropout=attn_dropout, dropout=dropout,
#                                                       activation=activation, res_attention=res_attention,
#                                                       pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
#         self.res_attention = res_attention
#
#     def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
#         output = src
#         scores = None
#         if self.res_attention:
#             for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#             return output
#         else:
#             for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#             return output
#
#
#
# class TSTEncoderLayer(nn.Module):
#     def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
#                  norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
#         super().__init__()
#         assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
#         d_k = d_model // n_heads if d_k is None else d_k
#         d_v = d_model // n_heads if d_v is None else d_v
#
#         # Multi-Head attention
#         self.res_attention = res_attention
#         self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
#
#         # Add & Norm
#         self.dropout_attn = nn.Dropout(dropout)
#         if "batch" in norm.lower():
#             self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
#         else:
#             self.norm_attn = nn.LayerNorm(d_model)
#
#         # Position-wise Feed-Forward
#         self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
#                                 get_activation_fn(activation),
#                                 nn.Dropout(dropout),
#                                 nn.Linear(d_ff, d_model, bias=bias))
#
#         # Add & Norm
#         self.dropout_ffn = nn.Dropout(dropout)
#         if "batch" in norm.lower():
#             self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
#         else:
#             self.norm_ffn = nn.LayerNorm(d_model)
#
#         self.pre_norm = pre_norm
#         self.store_attn = store_attn
#
#
#     def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
#
#         # Multi-Head attention sublayer
#         if self.pre_norm:
#             src = self.norm_attn(src)
#         ## Multi-Head attention
#         if self.res_attention:
#             src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#         else:
#             src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#         if self.store_attn:
#             self.attn = attn
#         ## Add & Norm
#         src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
#         if not self.pre_norm:
#             src = self.norm_attn(src)
#
#         # Feed-forward sublayer
#         if self.pre_norm:
#             src = self.norm_ffn(src)
#         ## Position-wise Feed-Forward
#         src2 = self.ff(src)
#         ## Add & Norm
#         src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
#         if not self.pre_norm:
#             src = self.norm_ffn(src)
#
#         if self.res_attention:
#             return src, scores
#         else:
#             return src
#
#
#
#
# class _MultiheadAttention(nn.Module):
#     def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
#         """Multi Head Attention Layer
#         Input shape:
#             Q:       [batch_size (bs) x max_q_len x d_model]
#             K, V:    [batch_size (bs) x q_len x d_model]
#             mask:    [q_len x q_len]
#         """
#         super().__init__()
#         d_k = d_model // n_heads if d_k is None else d_k
#         d_v = d_model // n_heads if d_v is None else d_v
#
#         self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
#
#         self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
#         self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
#         self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
#
#         # Scaled Dot-Product Attention (multiple heads)
#         self.res_attention = res_attention
#         self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)
#
#         # Poject output
#         self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
#
#
#     def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
#                 key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
#
#         bs = Q.size(0)
#         if K is None: K = Q
#         if V is None: V = Q
#
#         # Linear (+ split in multiple heads)
#         q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
#         k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
#         v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]
#
#         # Apply Scaled Dot-Product Attention (multiple heads)
#         if self.res_attention:
#             output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#         else:
#             output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#         # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]
#
#         # back to the original inputs dimensions
#         output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]    contiguous() 方法让 output 在内存中连续存储。这一步是为了使得 view 操作能够正确执行。
#         output = self.to_out(output)
#
#         if self.res_attention: return output, attn_weights, attn_scores
#         else: return output, attn_weights
#
#
# class _ScaledDotProductAttention(nn.Module):
#     r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
#     (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
#     by Lee et al, 2021)"""
#
#     def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
#         super().__init__()
#         self.attn_dropout = nn.Dropout(attn_dropout)
#         self.res_attention = res_attention
#         head_dim = d_model // n_heads
#         self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
#         self.lsa = lsa
#
#     def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
#         '''
#         Input shape:
#             q               : [bs x n_heads x max_q_len x d_k]
#             k               : [bs x n_heads x d_k x seq_len]
#             v               : [bs x n_heads x seq_len x d_v]
#             prev            : [bs x n_heads x q_len x seq_len]
#             key_padding_mask: [bs x seq_len]
#             attn_mask       : [1 x seq_len x seq_len]
#         Output shape:
#             output:  [bs x n_heads x q_len x d_v]
#             attn   : [bs x n_heads x q_len x seq_len]
#             scores : [bs x n_heads x q_len x seq_len]
#         '''
#
#         # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
#         attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]
#
#         # Add pre-softmax attention scores from the previous layer (optional)
#         if prev is not None: attn_scores = attn_scores + prev
#
#         # Attention mask (optional)
#         if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
#             if attn_mask.dtype == torch.bool:
#                 attn_scores.masked_fill_(attn_mask, -np.inf)
#             else:
#                 attn_scores += attn_mask
#
#         # Key padding mask (optional)
#         if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
#             attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
#
#         # normalize the attention weights
#         attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
#         attn_weights = self.attn_dropout(attn_weights)
#
#         # compute the new values given the attention weights
#         output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]
#
#         if self.res_attention: return output, attn_weights, attn_scores
#         else: return output, attn_weights
#
