import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_text(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vocab_size, self.embedding_dim, self.n_hidden, self.n_out, self.num_layers = args.vocab_size, args.embedding_dim, args.n_hidden, args.n_out, args.num_layers
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        #self.emb.weight = nn.Parameter(args.weights, requires_grad=False)
        self.gru = nn.LSTM(self.embedding_dim, self.n_hidden, self.num_layers, bidirectional = False)
        self.out = nn.Linear(self.n_hidden*3, self.n_out)

    def forward(self, seq, lengths):
        #print(seq.size())
        self.h, self.c = self.init_hidden(seq.size(1))
        embs = self.emb(seq)
        #print(embs.size())
        embs = pack_padded_sequence(embs, lengths)
        lstm_out, (self.h, self.c) = self.gru(embs, (self.h, self.c))
        lstm_out, lengths = pad_packed_sequence(lstm_out)
        #print(gru_out.size())
        avg_pool = F.adaptive_avg_pool1d(lstm_out.permute(1, 2, 0), 1).view(seq.size(1), -1)
        #print('Adaptive avg pooling', avg_pool.size())

        # adaptive avg pooling by hand
        # taking the sum along the batch axis and dividing by the corresponding lengths to get the actual mean
        #avg_pool_byhand = torch.sum(gru_out, dim=0) / Variable(torch.FloatTensor(lengths).view(-1, 1))
        #print('By hand Adaptive avg pooling', avg_pool_byhand)

        max_pool = F.adaptive_max_pool1d(lstm_out.permute(1, 2, 0), 1).view(seq.size(1), -1)
        #print('Adaptive max pooling', max_pool.size())

        # adaptive max pooling by hand
        # collect all the non padded elements of the batch and then take max of them
        # max_pool_byhand = torch.cat([torch.max(i[:l], dim=0)[0].view(1, -1) for i, l in zip(gru_out.permute(1, 0, 2), lengths)], dim=0)
        #print('By hand Adaptive max pooling', max_pool_byhand)

        # outp = self.out(torch.cat([self.h[-1],avg_pool,max_pool],dim=1))
        outp = self.out(torch.cat([self.c[-1], avg_pool, max_pool], dim=1))
        return F.log_softmax(outp, dim=-1)

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((self.num_layers, batch_size, self.n_hidden),device=torch.device("cuda")))
        c = Variable(torch.zeros((self.num_layers, batch_size, self.n_hidden),device=torch.device("cuda")))
        return h,c