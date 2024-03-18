import numpy as np
import torch
import torch as t
from torch.utils.data import DataLoader, Dataset
from torch.nn import *

class LTP(Module):

    def __init__(self, args):
        super(LTP, self).__init__()
        self.rank = args.dim
        self.window = args.window
        self.device = args.device
        self.user_embeds = Embedding(args.users, self.rank)
        self.item_embeds = Embedding(args.items, self.rank)
        self.time_embeds = Embedding(args.times, self.rank)
        self.lstm = LSTM(self.rank, self.rank, batch_first=False)
        self.rainbow = t.arange(-self.window + 1, 1).reshape(1, -1).to(self.device)
        self.attn = Sequential(Linear(2 * self.rank, 1), Tanh())
        self.user_linear = Linear(self.rank, self.rank)
        self.item_linear = Linear(self.rank, self.rank)
        self.time_linear = Linear(self.rank, self.rank)
        self.simple = False

    def next(self):
        pass
        # init.normal_(self.user_embed.weight)
        # init.normal_(self.item_embed.weight)
        # init.normal_(self.time_embed.weight)

    def to_seq_id(self, tids):
        tids = tids.reshape(-1, 1).repeat(1, self.window)
        tids += self.rainbow
        tids = tids.relu().permute(1, 0)
        return tids


    def forward(self, user, item, time, return_embeds=False):
        user_embeds = self.get_embeddings(user, "user").to(self.device)
        item_embeds = self.get_embeddings(item, "item").to(self.device)
        time_embeds = self.get_embeddings(time, "time").to(self.device)
        if not return_embeds:
            return self.get_score(user_embeds, item_embeds, time_embeds)
        else:
            return self.get_score(user_embeds, item_embeds, time_embeds), user_embeds, item_embeds, time_embeds

    def get_score(self, user_embeds, item_embeds, time_embeds):
        if self.simple:
            return self.get_final_score(user_embeds, item_embeds, time_embeds)
        # Interaction Modules
        user_embeds = self.user_linear(user_embeds)
        item_embeds = self.item_linear(item_embeds)
        time_embeds = self.time_linear(time_embeds)
        raw_score = t.sum(user_embeds * item_embeds * time_embeds, dim=-1)
        pred = raw_score.sigmoid()
        return pred

    def get_embeddings(self, idx, select):
        if self.simple:
            return self.get_final_embeddings(idx, select)

        if select == "user":
            return self.user_embeds(idx)

        elif select == "item":
            return self.item_embeds(idx)

        elif select == "time":
            # Read Time Embeddings
            time_embeds = self.time_embeds(self.to_seq_id(idx))
            outputs, (hs, cs) = self.lstm.forward(time_embeds)

            # Attention [seq_len, batch, dim] -> [seq_len, batch, 1]
            hss = hs.repeat(self.window, 1, 1)
            attn = self.attn(t.cat([outputs, hss], dim=-1))
            time_embeds = t.sum(attn * outputs, dim=0)
            return time_embeds
        else:
            raise NotImplementedError("Unknown select type: {}".format(select))


    def get_final_embeddings(self, idx, select):
        if select == "user":
            user_embeds = self.user_embeds(idx)
            return self.user_linear(user_embeds)

        elif select == "item":
            item_embeds = self.item_embeds(idx)
            return self.item_linear(item_embeds)

        elif select == "time":
            # Read Time Embeddings
            time_embeds = self.time_embeds(self.to_seq_id(idx))
            outputs, (hs, cs) = self.lstm.forward(time_embeds)

            # Attention [seq_len, batch, dim] -> [seq_len, batch, 1]
            hss = hs.repeat(self.window, 1, 1)
            attn = self.attn(t.cat([outputs, hss], dim=-1))
            time_embeds = t.sum(attn * outputs, dim=0)
            time_embeds = self.time_linear(time_embeds)
            return time_embeds
        else:
            raise NotImplementedError("Unknown select type: {}".format(select))

    def get_final_score(self, user_embeds, item_embeds, time_embeds):
        pred = t.sum(user_embeds * item_embeds * time_embeds, dim=-1).sigmoid()
        return pred
