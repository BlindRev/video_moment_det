import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from ytbHighlight_testData_videoBank import ytbHighlight_testData
from opts import parse_opts
import time
#from utils import Logger
#from utils import Bar
import os
from sklearn.metrics import average_precision_score
import numpy as np
import math

def pairwise_loss(s_i, s_j, sigma=1):
    C = torch.log1p(torch.exp(-sigma * (s_i - s_j)))
    loss = C.mean()
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        #if dropout is not None:
        #    attention = dropout(attention)
        return attention.matmul(value), attention 


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=False,
                 activation= None, #F.relu
                 dropout = 0.1):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.attention_head_size = int(in_features / head_num)
        self.activation = None
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        self.layer_norm = nn.LayerNorm(in_features)
        
        nn.init.normal_(self.linear_q.weight, mean=0, std=np.sqrt(2.0 / (in_features)))
        nn.init.normal_(self.linear_k.weight, mean=0, std=np.sqrt(2.0 / (in_features)))
        nn.init.normal_(self.linear_v.weight, mean=0, std=np.sqrt(2.0 / (in_features)))
        nn.init.xavier_normal_(self.linear_o.weight)

    def forward(self, q, k, v, mask=None):
        residual = q
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        context_layer, att = ScaledDotProductAttention()(q, k, v, mask, self.dropout)
        context_layer = self._reshape_from_batches(context_layer)
        att = torch.mean(att, dim=0)
        
        
        
        context_layer = self.dropout(self.linear_o(context_layer))
        if self.activation is not None:
            context_layer = self.activation(context_layer)
        context_layer = self.layer_norm(context_layer + residual)
        return context_layer, att

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)
    
    
class RankNet(nn.Module):
    def __init__(self, num_feature, head_num=8, comb_mode = 'concat'):
        super(RankNet, self).__init__()
        self.comb_mode = comb_mode
        self.attn = MultiHeadAttention(num_feature, head_num)
        if comb_mode == 'concat':
            in_fea_dim = 2 * num_feature
        if comb_mode == 'context_only':
            in_fea_dim = num_feature
        self.score = nn.Sequential(
            nn.Linear(in_fea_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.output_sig = nn.Sigmoid()

    def forward(self, input1, input2):
        ##concat input1 and input2
        bs, dim = input1.size()
        data_concat = (torch.cat((input1, input2), 0)).view(1, 2*bs, dim)
        context_fea, att = self.attn(data_concat,data_concat,data_concat,None)       
        context_fea1, context_fea2 = torch.chunk(torch.squeeze(context_fea), 2 , dim=0)
        if self.comb_mode == 'concat':
            x1 = torch.cat((input1, context_fea1), 1)
            x2 = torch.cat((input2, context_fea2), 1)
            s_i = self.score(x1)
            s_j = self.score(x2)
        if self.comb_mode == 'context_only':
            s_i = self.score(context_fea1)
            s_j = self.score(context_fea2)
        
        out = self.output_sig(s_i-s_j)
        return out, att
    
    def predict(self, input_):
        x = input_
        context_fea, _ = self.attn(x,x,x,None)
        context_fea = torch.squeeze(context_fea)
        if self.comb_mode == 'concat':
            fea = torch.cat((torch.squeeze(input_), context_fea), 1)
            s = self.score(fea)
        if self.comb_mode == 'context_only':
            s = self.score(context_fea)
        return s

    
def find_matching_snippet(clip, snippets):
    indx = []
    for i, snippet in enumerate(snippets):
        clip_frames = list(range(int(clip[0]), int(clip[1])+1))
        intersection = len([f for f in snippet if f in clip_frames])
        if intersection > 0.7 * len(snippet):
            indx.append(i)
    return indx

def get_clip_scores(clips, snippet_scores):
    clip_scores = []
    bad_indx = []
    snippets = [snippet_scores[i][0] for i in range(len(snippet_scores))]    
    for ind, clip in enumerate(clips):
        indx = find_matching_snippet(clip, snippets)
        if len(indx) == 0:
            bad_indx.append(ind)
        else:
            scores = [snippet_scores[i][1] for i in indx]
            clip_scores.append(sum(scores)/len(scores))
    return clip_scores, bad_indx


def valid(epoch, valid_loader, model,criterion):
    end_time = time.time()
    model.eval()
    losses = AverageMeter()
    end_time = time.time()
    bar = Bar('Processing', max=len(valid_loader))
    for i, (data1, data2) in enumerate(valid_loader):
        data_time.update(time.time() - end_time)
        data1 = Variable(data1.cuda())
        data2 = Variable(data2.cuda())
        pred, _ = model(data1, data2)
        loss = criterion(pred, torch.from_numpy(np.ones(shape=(data1.size()[0], 1))).float().cuda())
        losses.update(loss.data.item(), data1.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        bar.suffix  = 'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | Valid Loss {loss:.4f}'.format(
                  epoch + 1,
                  i + 1,
                  len(valid_loader),
                  batch_time=batch_time,
                  loss=losses.avg)
        bar.next()
    bar.finish()
    return losses.avg

def test(test_loader, rank_model):
    print('predicting scores for snippets...')
    rank_model.eval()
    pred_scores = {}
    with torch.no_grad():
        for fea, vids, frames in test_loader:
            #print(vids, len(vids), fea.shape)
            #break
            scores = rank_model.predict(fea.cuda())
            for i, vid_ in enumerate(vids):
                vid = vid_[0]
                if vid not in pred_scores:
                    pred_scores[vid] = []
                else:
                    frame_ids = [ int(frames[i][j].cpu()) for j in range(16)]
                    pred_scores[vid].append([frame_ids , float(scores[i].cpu())])
    print('calculate MAP...')
    annotate = test_set.annotate
    map_total = 0
    num = 0
    for vid in annotate.keys():
        label = annotate[vid][1]
        preds , bad_indx = get_clip_scores(annotate[vid][0], pred_scores[vid])
        if len(bad_indx) > 0:
            tmp = [i for j, i in enumerate(label) if j not in bad_indx]
            label = tmp    
        y_true = np.array(label)
        indx = np.where(y_true == -1)
        y_true[indx] = 0
        y_scores = np.array(preds)
        map_per_vid = average_precision_score(y_true, y_scores)
        map_total+=map_per_vid
        num += 1
    print('{}_MAP:{}'.format(opt.domain, map_total/num) )
    return map_total/num


import os
if __name__ == '__main__':
    opt = parse_opts()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    epochs = opt.epochs
    batch_size = opt.batch_size
    rank_model = RankNet(num_feature=512, head_num=8)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(rank_model.parameters())
    
    if opt.test:
        checkpoint = torch.load(opt.pretrain_path)
        rank_model.load_state_dict(checkpoint['state_dict'])
        rank_model.cuda()
        rank_model.eval()
        print('Loading test data')
        test_set = ytbHighlight_testData(opt.ytbData, opt.domain, 'test')
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
   
        test(test_loader, rank_model)
