import pandas as pd
import numpy as np
import os
import math
from typing import Optional
from collections import OrderedDict
from tqdm import trange, tqdm
from dotenv import load_dotenv

import logging
import torch
from torch import nn, Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error

from ..dataproc import DataProc

# .env 파일 로드
load_dotenv()

# 환경 변수 읽기
path = os.getenv("PATH")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 활성화 함수 반환
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

# Positional Encoding
def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src

class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output1 = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output1 = self.output_layer(output1)  # (batch_size, seq_length, feat_dim)

        return output1, output
    
class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.
        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered
        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """
        # print(y_pred.shape, mask.shape)# torch.Size([32, 3]) torch.Size([32, 28, 3])
        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)

# Transformer AutoEncoder 실행
class TAE:
    def __init__(self, dp:DataProc, num_epochs=200, batch_size=128) -> None:

        # 데이터 처리용 클래스
        self.dp = dp
        self.train_loader = DataLoader(dataset=self.dp.ds_train, batch_size=batch_size, drop_last=True, shuffle=True)
        self.test_loader = DataLoader(dataset=self.dp.ds_test, batch_size=batch_size, drop_last=False, shuffle=False)
        self.cluster_loader = DataLoader(dataset=self.dp.ds_cluster, batch_size=batch_size, drop_last=False, shuffle=False)

        #  Hyper-parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self):
        model = TSTransformerEncoder(feat_dim=self.dp.data_train.shape[2], max_len=self.dp.data_train.shape[1], 
                                     d_model=64, n_heads=8, num_layers=1, dim_feedforward=256, pos_encoding='learnable')
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters())

        loss_module=MaskedMSELoss()
        loss_list = []
        val_list = []
        valmask_list = []
        val_output = []
        val_true = []
        criterion = nn.MSELoss()
        for epoch in tqdm(range(self.num_epochs)):
            epoch_loss = 0  # total loss of epoch
            total_active_elements = 0  # total unmasked elements in epoch
            for i, batch in enumerate(self.train_loader):

                X, targets, target_masks = batch # X는 mask되지 않은 값
                target_masks = target_masks.to(device) # 1s: mask and predict, 0s: unaffected input (ignore) // noise_mask 에서 옴
                X = X.to(device)
                targets = torch.tensor(targets.to(device), dtype = torch.float32) # mask 되지 않은 값
                X = X * target_masks # mask 된 input
                X = torch.tensor(X, dtype = torch.float32)
                predictions, _ = model(X)  # (batch_size, padded_length, feat_dim)

                # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
                loss = loss_module(predictions, targets, ~target_masks)  # ~target_mask를 prediction, target에 역으로 곱해서 mask한 값만 loss 계산함
                total_loss = loss
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()
                total_active_elements += 1
                epoch_loss += total_loss.item()
            with torch.no_grad():
                val_loss = 0.0
                valmask_loss = 0.0
                for i, batch in enumerate(self.test_loader):
                    X, targets, target_masks = batch # X는 mask되지 않은 값
                    target_masks = target_masks.to(device) # 1s: mask and predict, 0s: unaffected input (ignore) // noise_mask 에서 옴
                    X = X.to(device)
                    targets = torch.tensor(targets.to(device), dtype = torch.float32) # mask 되지 않은 값
                    X = X * target_masks # mask 된 input
                    X = torch.tensor(X, dtype = torch.float32)
                    predictions, _ = model(X)
                    val_loss += criterion(predictions, targets.float()).item()
                    valmask_loss += loss_module(predictions, targets, ~target_masks)
                val_list.append(val_loss)
                valmask_list.append(valmask_loss)
                if epoch == self.num_epochs-1:
                    val_true.append(targets.cpu())
                    val_output.append(predictions.cpu())
            epoch_loss = epoch_loss / total_active_elements
            loss_list.append(epoch_loss)
        print(epoch_loss)

    def predict(self):
        pass