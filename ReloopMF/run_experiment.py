# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import torch
import argparse

from tqdm import *

from utils.config import get_config
from utils.dataloader import get_dataloaders
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed

global log

torch.set_default_dtype(torch.float32)


class experiment:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def load_data(args):
        string = args.path + '/' + args.dataset + 'Matrix' + '.txt'
        tensor = np.loadtxt(open(string, 'rb'))
        return tensor

    @staticmethod
    def preprocess_data(data, args):
        data[data == -1] = 0
        return data



# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.data = exper_type.load_data(args)
        self.data = exper_type.preprocess_data(self.data, args)
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = self.get_train_valid_test_dataset(self.data, args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_tensor, self.valid_tensor, self.test_tensor, exper_type, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')

    def get_dataset(self, train_tensor, valid_tensor, test_tensor, exper_type, args):
        return (
            TensorDataset(train_tensor, exper_type, args),
            TensorDataset(valid_tensor, exper_type, args),
            TensorDataset(test_tensor, exper_type, args)
        )

    def get_train_valid_test_dataset(self, tensor, args):
        quantile = np.percentile(tensor, q=100)
        # tensor[tensor > quantile] = 0
        tensor = tensor / (np.max(tensor))  # 如果数据有分布偏移，记得处理数据
        trainsize = int(np.prod(tensor.size) * args.density)
        validsize = int((np.prod(tensor.size)) * 0.05)
        rowIdx, colIdx = tensor.nonzero()
        p = np.random.permutation(len(rowIdx))
        rowIdx, colIdx = rowIdx[p], colIdx[p]
        trainRowIndex = rowIdx[:trainsize]
        trainColIndex = colIdx[:trainsize]
        traintensor = np.zeros_like(tensor)
        traintensor[trainRowIndex, trainColIndex] = tensor[trainRowIndex, trainColIndex]
        validStart = trainsize
        validRowIndex = rowIdx[validStart:validStart + validsize]
        validColIndex = colIdx[validStart:validStart + validsize]
        validtensor = np.zeros_like(tensor)
        validtensor[validRowIndex, validColIndex] = tensor[validRowIndex, validColIndex]
        testStart = validStart + validsize
        testRowIndex = rowIdx[testStart:]
        testColIndex = colIdx[testStart:]
        testtensor = np.zeros_like(tensor)
        testtensor[testRowIndex, testColIndex] = tensor[testRowIndex, testColIndex]
        return traintensor, validtensor, testtensor, quantile


class TensorDataset(torch.utils.data.Dataset):

    def __init__(self, tensor, exper, args):
        self.tensor = tensor
        self.indices = self.get_pytorch_index(tensor)

    def __getitem__(self, idx):
        output = self.indices[idx, :-1]  # 去掉最后一列
        inputs = tuple(torch.as_tensor(output[i]).long() for i in range(output.shape[0]))
        value = torch.as_tensor(self.indices[idx, -1])  # 最后一列作为真实值
        return inputs, value

    def __len__(self):
        return self.indices.shape[0]

    def get_pytorch_index(self, data):
        userIdx, servIdx = data.nonzero()
        values = data[userIdx, servIdx]
        idx = torch.as_tensor(np.vstack([userIdx, servIdx, values]).T)
        return idx


class ExternalAttention(torch.nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = torch.nn.Linear(d_model, S, bias=False)
        self.mv = torch.nn.Linear(S, d_model, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        queries = queries.to(torch.float32)
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model
        return out


class Attention(torch.nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.input_dim = dim * 2
        self.attention = ExternalAttention(self.dim, 32)


    def forward(self, a, b):
        embeds = torch.hstack([a.unsqueeze(1), b.unsqueeze(1)])
        embeds = self.attention(embeds).reshape(len(embeds), -1)
        return embeds


import faiss
class CorrectError(torch.nn.Module):
    def __init__(self, args):
        super(CorrectError, self).__init__()
        self.args = args
        self.k = args.k
        self.tau = 0.1
        self.memory_embeds = []  # 存储嵌入向量
        self.true_values = []  # 存储对应的预测值
        self.index = faiss.IndexLSH(self.args.dimension * 2, 32)
        self.build = False

    def reset_memory(self):
        self.build = False
        self.memory_embeds = []
        self.true_values = []  # 存储对应的预测值
        self.index = faiss.IndexLSH(self.args.dimension * 2, 32)

    def add_memory(self, embeds, true_value):
        for i in range(len(embeds)):
            self.memory_embeds.append(embeds[i])
            self.true_values.append(true_value[i])

    def get_topk_embeds(self, h_query):
        if self.args.device == 'cuda':
            h_query = h_query.cpu().detach().numpy().astype('float32')
        else:
            h_query = h_query.detach().numpy().astype('float32')

        D, topk_indices = self.index.search(h_query, self.k)
        topk_indices = torch.tensor(topk_indices, dtype=torch.long)
        # print(topk_indices.shape)
        # topk_values = torch.as_tensor(self.pred_values)

        true_values = torch.stack(self.true_values)
        all_topk_values = []
        for i in range(len(topk_indices)):
            now_topk_values = []
            for j in range(len(topk_indices[i])):
                now_topk_values.append(true_values[topk_indices[i][j]])
            all_topk_values.append(now_topk_values)
        topk_values = torch.as_tensor(all_topk_values)
        return 1, topk_values

    def forward(self, h_query):
        if not self.build:
            embeds = torch.stack(self.memory_embeds).cpu().detach().numpy()
            self.index.train(embeds)
            self.index.add(embeds)
            self.build = True
        topk_embeds, topk_true_values = self.get_topk_embeds(h_query)
        y_bar = torch.mean(topk_true_values, dim=-1)
        return y_bar


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.user_num = args.user_num
        self.serv_num = args.serv_num

        # NeuCF
        self.dropout = 0.10
        self.num_layers = 2
        self.dimension = args.dimension
        self.dimension_gmf = args.dimension
        self.dimension_mlp = args.dimension * (2 ** (self.num_layers - 1))
        self.embed_user_GMF = torch.nn.Embedding(self.user_num, self.dimension_gmf)
        self.embed_user_MLP = torch.nn.Embedding(self.user_num, self.dimension_mlp)
        self.embed_item_GMF = torch.nn.Embedding(self.serv_num, self.dimension_gmf)
        self.embed_item_MLP = torch.nn.Embedding(self.serv_num, self.dimension_mlp)

        MLP_modules = []
        for i in range(self.num_layers):
            input_size = self.dimension * (2 ** (self.num_layers - i))
            MLP_modules.append(torch.nn.Dropout(p=self.dropout))
            MLP_modules.append(torch.nn.Linear(input_size, input_size // 2))
            MLP_modules.append(torch.nn.ReLU())
        self.MLP_layers = torch.nn.Sequential(*MLP_modules)
        self.predict_layer = torch.nn.Linear(self.dimension * 2, 1)

        # Att
        self.dim = args.dimension
        self.user_embedding = torch.nn.Embedding(self.user_num, self.dim)
        self.serv_embedding = torch.nn.Embedding(self.serv_num, self.dim)
        self.attention = Attention(self.dim)
        self.input_dim = self.dim * 2
        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim // 2),  # FFN
            torch.nn.LayerNorm(self.input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(self.input_dim // 2, self.input_dim // 2),  # FFN
            torch.nn.LayerNorm(self.input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(self.input_dim // 2, 1)  # y
        )
        # Correction
        self.Lambda = torch.nn.Parameter(torch.randn(1))
        self.correct_layer = CorrectError(args)
        # self.correct_layer = MemoryModule(args)

    def NeuCF(self, userIdx, servIdx):
        user_embed = self.embed_user_GMF(userIdx)
        embed_user_MLP = self.embed_user_MLP(userIdx)
        item_embed = self.embed_item_GMF(servIdx)
        embed_item_MLP = self.embed_item_MLP(servIdx)
        gmf_output = user_embed * item_embed
        mlp_input = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        mlp_output = self.MLP_layers(mlp_input)
        embeds = torch.cat((gmf_output, mlp_output), -1)
        y_base = self.predict_layer(embeds)
        # embeds = self.attention(gmf_output, mlp_output)
        # y_base = self.interaction(embeds)
        return y_base, embeds

    def AttCF(self, userIdx, servIdx):
        user_embeds = self.user_embedding(userIdx)
        serv_embeds = self.serv_embedding(servIdx)
        embeds = self.attention(user_embeds, serv_embeds)
        y_base = self.interaction(embeds)
        return y_base, embeds

    def forward(self, inputs, test=False):
        userIdx, servIdx = inputs
        y_base, embeds = self.NeuCF(userIdx, servIdx)
        # y_base, embeds = self.AttCF(userIdx, servIdx)
        return embeds, y_base.flatten()

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=0.01)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        # Reset the memory
        for train_Batch in dataModule.train_loader:
            inputs, value = train_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            embeds, y_base = self.forward(inputs, False)

            # Memory the embeds
            # 存储训练集的True label
            # self.correct_layer.add_memory(embeds, y_base)

            loss = self.loss_function(y_base, value)
            optimizer_zero_grad(self.optimizer)
            loss.backward()
            optimizer_step(self.optimizer)
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def valid_one_epoch(self, dataModule):
        writeIdx = 0
        val_loss = 0.
        crt_preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        raw_preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        for valid_Batch in (dataModule.valid_loader):
            inputs, value = valid_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            _, raw_pred = self.forward(inputs)
            # Correction
            if self.args.check:
                embeds, y_base = self.forward(inputs, True)
                self.correct_layer.add_memory(embeds, value)
                y_bar = self.correct_layer(embeds)
                crt_pred = 0.1 * y_bar + 0.9 * y_base
                crt_pred = crt_pred.flatten()
            val_loss += self.loss_function(crt_pred, value).item()
            crt_preds[writeIdx: writeIdx + len(crt_pred)] = crt_pred
            raw_preds[writeIdx:writeIdx + len(crt_pred)] = raw_pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(crt_pred)
        self.scheduler.step(val_loss)
        crt_valid_error = ErrorMetrics(reals * dataModule.max_value, crt_preds * dataModule.max_value)
        raw_valid_error = ErrorMetrics(reals * dataModule.max_value, raw_preds * dataModule.max_value)
        return crt_valid_error, raw_valid_error

    def test_one_epoch(self, dataModule):
        # print(len(self.correct_layer.memory_embeds))
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        for test_Batch in tqdm(dataModule.test_loader):
            inputs, value = test_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            pred = self.forward(inputs)

            # Correction
            if self.args.check:
                embeds, y_base = self.forward(inputs, True)
                y_bar = self.correct_layer(embeds)
                pred = 0.1 * y_bar + 0.9 * y_base
                pred = pred.flatten()
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return test_error



def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(args)
    monitor = EarlyStopping(args)

    # Setup training tool
    model.setup_optimizer(args)
    model.max_value = datamodule.max_value
    train_time = []
    # for epoch in trange(args.epochs, disable=not args.program_test):
    for epoch in range(args.epochs):
        model.correct_layer.reset_memory()
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        crt_valid_error, raw_valid_error = model.valid_one_epoch(datamodule)
        monitor.track_one_epoch(epoch, model, crt_valid_error)
        train_time.append(time_cost)
        log.only_print('Raw Error')
        log.show_epoch_error(runId, epoch, epoch_loss, raw_valid_error, train_time)
        log.only_print('Crt Error')
        log.show_epoch_error(runId, epoch, epoch_loss, crt_valid_error, train_time)
        print('-' * 80)
        if monitor.early_stop:
            break
    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    results = model.test_one_epoch(datamodule)
    log.show_test_error(runId, monitor, results, sum_time)
    return {
        'MAE': results["MAE"],
        'RMSE': results["RMSE"],
        'NMAE': results["NMAE"],
        'NRMSE': results["NRMSE"],
        'TIME': sum_time,
    }, results['Acc']


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)
    for runId in range(args.rounds):
        runHash = int(time.time())
        results, acc = RunOnce(args, runId, runHash, log)
        for key in results:
            metrics[key].append(results[key])
        for key, item in zip(['Acc1', 'Acc5', 'Acc10'], [0, 1, 2]):
            metrics[key].append(acc[item])
    log('*' * 20 + 'Experiment Results:' + '*' * 20)
    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')
    if args.record:
        log.save_result(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20 + '\n')
    return metrics


if __name__ == '__main__':
    args = get_config()
    set_settings (args)
    log = Logger(args)
    args.log = log
    log(str(args))
    RunExperiments(log, args)