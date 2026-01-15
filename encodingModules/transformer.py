import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import math, sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer

# 경로 추가
path = '/workspace/AMI/jspark'
sys.path.append(path)

if __name__ == "__main__":
    from dataproc import DataProc
    from tae_modules import TSTransformerEncoder
else:
    from dataproc import DataProc
    from .tae_modules import TSTransformerEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerDataset(Dataset):
    def __init__(self, X, mask):
        self.X = X
        self.target = X
        self.mask = mask
        # self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.target[index], self.mask[index]

class TransformerDataProc(DataProc):
    # LSTM AutoEncoder 전용 데이터셋 생성
    def transformer_dataset(self):
        # 전처리
        self.preprocess()
        
        data_train = []
        for user in range(self.elec_total.shape[1]):
            for index in range(0,self.elec_total.shape[0]//(28*2)-2):
                window = []
                for period in range(index*28*2,(index+1)*28*2):
                    window.append([self.energy_train[0][period,user],self.energy_train[1][period,user],self.energy_train[2][period,user]])
                data_train.append(window)

        data_test = []
        for user in range(self.elec_total.shape[1]):
            for index in range(0,1):
                window = []
                for period in range(index*28*2,(index+1)*28*2):
                    window.append([self.energy_test[0][period,user],self.energy_test[1][period,user],self.energy_test[2][period,user]])
                data_test.append(window)
                
        data_cluster = []
        for user in range(self.elec_total.shape[1]):
            for index in range(0,1):
                window = []
                for period in range(index*28*2,(index+1)*28*2):
                    window.append([self.energy_cluster[0][period,user],self.energy_cluster[1][period,user],self.energy_cluster[2][period,user]])
                data_cluster.append(window)

        # dataset
        data_train = np.array(data_train)
        data_test = np.array(data_test)
        data_cluster = np.array(data_cluster)
        self.data_train = data_train
        self.data_test = data_test
        self.data_cluster = data_cluster

        # masking
        data_train_mask = []
        for i in range(data_train.shape[0]):
            data_train_mask.append(noise_mask(data_train[i]))
            
        data_test_mask = []
        for i in range(data_test.shape[0]):
            data_test_mask.append(noise_mask(data_test[i]))
            
        data_cluster_mask = []
        for i in range(data_cluster.shape[0]):
            data_cluster_mask.append(noise_mask(data_cluster[i]))

        data_train_mask = np.array(data_train_mask)
        data_test_mask = np.array(data_test_mask)
        data_train_mask = np.array(data_train_mask)

        ds_train = TransformerDataset(data_train, data_train_mask)
        ds_test = TransformerDataset(data_test, data_test_mask)
        ds_cluster = TransformerDataset(data_cluster, data_cluster_mask)

        batch_size = 128
        self.train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, drop_last=True, shuffle=True)
        self.test_loader = DataLoader(dataset=ds_test, batch_size=batch_size, drop_last=False, shuffle=False)
        self.cluster_loader = DataLoader(dataset=ds_cluster, batch_size=batch_size, drop_last=False, shuffle=False)

def TAE(target, num): # target 부분 아직 덜 함
    dp = TransformerDataProc()
    dp.transformer_dataset()

    model = TSTransformerEncoder(feat_dim=dp.data_train.shape[2], max_len=dp.data_train.shape[1], d_model=64, n_heads=8, num_layers=1, dim_feedforward=256, pos_encoding='learnable')
    model.to(device)
    # optimizer = RAdam(model.parameters())
    optimizer = torch.optim.AdamW(model.parameters())

    # training loop
    model.train()
    num_epochs=200
    loss_module=MaskedMSELoss()
    loss_list = []
    val_list = []
    valmask_list = []
    val_output = []
    val_true = []
    criterion = nn.MSELoss()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(dp.train_loader):

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
            for i, batch in enumerate(dp.test_loader):
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
            if epoch == num_epochs-1:
                val_true.append(targets.cpu())
                val_output.append(predictions.cpu())
        epoch_loss = epoch_loss / total_active_elements
        loss_list.append(epoch_loss)
    print(epoch_loss)


    cluster_hidden = []
    with torch.no_grad():
        for i, batch in enumerate(dp.cluster_loader):
            X, _, _ = batch # X는 mask되지 않은 값
            X = X.to(device)
            X = torch.tensor(X, dtype = torch.float32)
            _, hidden = model(X)
            cluster_hidden.append(hidden.cpu().numpy())

    arr_hidden = []
    for i in range(9):
        for j in range(cluster_hidden[i].shape[0]):
            arr_hidden.append(cluster_hidden[i][j])
    arr_hidden = pd.DataFrame(np.reshape(arr_hidden,(1080,64*28*2)))


            
    test_hidden = []
    with torch.no_grad():
        for i, batch in enumerate(dp.test_loader):
            X, _, _ = batch # X는 mask되지 않은 값
            X = X.to(device)
            X = torch.tensor(X, dtype = torch.float32)
            _, hidden = model(X)
            test_hidden.append(hidden.cpu().numpy())


    arr_hidden_2 = []
    for i in range(9):
        for j in range(test_hidden[i].shape[0]):
            arr_hidden_2.append(test_hidden[i][j])
    arr_hidden_2 = pd.DataFrame(np.reshape(arr_hidden_2,(1080,64*28*2)))

    arr_hidden.index = dp.elec_total.columns
    arr_hidden_2.index = dp.elec_total.columns
    arr_hidden.to_csv(f'{path}/hidden_layers/transformer_hidden_cluster_{target}_12H_{num}.csv')
    arr_hidden_2.to_csv(f'{path}/hidden_layers/transformer_hidden_test_{target}_12H_{num}.csv')
    torch.save(model, f'{path}/hidden_layers/transformer_{target}_12H_{num}.pt')

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
    
def noise_mask(X, masking_ratio=0.15, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

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
    
class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        # self.print_interval = print_interval
        # self.printer = utils.Printer(console=console)

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

class UnsupervisedRunner(BaseRunner):

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks = batch # X는 mask되지 않은 값
            target_masks = target_masks.to(self.device) # 1s: mask and predict, 0s: unaffected input (ignore) // noise_mask 에서 옴
            X = X.to(self.device)
            targets = targets.to(self.device) # mask 되지 않은 값
            X = X * target_masks # mask 된 input
            X = torch.tensor(X, dtype = torch.float32)
            targets = torch.tensor(targets, dtype = torch.float32)
            predictions = self.model(X)  # (batch_size, padded_length, feat_dim)
            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            loss = self.loss_module(predictions, targets, target_masks)  # ~target_mask를 prediction, target에 역으로 곱해서 mask한 값만 loss 계산함
            # batch_loss = torch.tensor(loss,dtype = torch.float32)
            # mean_loss = batch_loss #/ len(loss)  # mean loss (over active elements) used for optimization
            total_loss = loss
            # if self.l2_reg:
            #     total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            # else:
            #     total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            # total_loss=total_loss.requires_grad_(True)
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            # metrics = {"loss": mean_loss.item()}
            # if i % self.print_interval == 0:
            #     ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            #     self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_active_elements += 1#len(loss)
                # epoch_loss += batch_loss#.item()  # add total loss of batch 일단 주석처리(batch_loss 생성부분이 이미 주석처리되어 있어서 epoch_loss도 필요없다고 생각중)

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        val_output = []
        val_true = []
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks = batch # X는 mask되지 않은 값
            target_masks = target_masks.to(self.device) # 1s: mask and predict, 0s: unaffected input (ignore) // noise_mask 에서 옴
            X = X.to(self.device)
            targets = targets.to(self.device) # mask 되지 않은 값
            X = X * target_masks # mask 된 input
            X = torch.tensor(X, dtype = torch.float32)
            targets = torch.tensor(targets, dtype = torch.float32)
            
            
            predictions = self.model(X)  # (batch_size, padded_length, feat_dim)
            val_true.append(targets.cpu())
            val_output.append(predictions.cpu())
            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            
            batch_loss = loss.cpu().item()
            # mean_loss = batch_loss #/ len(loss)  # mean loss (over active elements) used for optimization the batch

            # if keep_all:
            #     per_batch['target_masks'].append(target_masks.cpu().numpy())
            #     per_batch['targets'].append(targets.cpu().numpy())
            #     per_batch['predictions'].append(predictions.cpu().numpy())
            #     per_batch['metrics'].append([loss.cpu().numpy()])
            #     per_batch['IDs'].append(IDs)

            # metrics = {"loss": mean_loss}
            # if i % self.print_interval == 0:
            #     ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            #     self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            # total_active_elements += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        # epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        return self.epoch_metrics, val_true, val_output