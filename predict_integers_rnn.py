# https://github.com/hughperkins/pub-prototyping/blob/master/py/pytorch/predict_integers_rnn.py

import torch
from torch import nn, optim
import torch.nn.functional as F
from numSeqPredictor import get_each_num

vocab_size = 60
embedding_size = 32

"""
inputs discrete, bunch of integers
=> embedding  nn.Embedding
=> RNN        nn.RNN
=> e2v        nn.Linear
=> outputs    .max
=> discrete predictions
"""

embedding = nn.Embedding(vocab_size, embedding_size)
rnn = nn.LSTM(embedding_size, embedding_size, batch_first=True)
e2v = nn.Linear(embedding_size, vocab_size)

final_pred_seq = []

for i in range(0, 7):
    integer_sequence = get_each_num()[i]
    # integer_sequence.append(0)
    inputs = integer_sequence[:-1]
    gold_outputs = integer_sequence[1:]

    batch_size = 1
    seq_len = len(gold_outputs)

    print('inputs', inputs)
    print('gold_outputs', gold_outputs)

    inputs_t = torch.tensor([inputs])
    opt = optim.Adam(lr=0.005, params=list(
        embedding.parameters()) + list(rnn.parameters()) + list(e2v.parameters()))

    i = 0
    for epoch in range(1000):
        i += 1
        x = embedding(inputs_t.squeeze())
        emb_out, (h, c) = rnn(x)
        outputs = e2v(emb_out)

        outputs_flat = outputs.view(batch_size * seq_len, vocab_size)
        gold_outputs_flat = torch.tensor([gold_outputs]).view(
            batch_size * seq_len
        )
        loss = F.cross_entropy(outputs_flat, gold_outputs_flat)
        # if i % 1 == 0:
        #     print("i=", i)
        #     print('loss %.4f' % loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        _, preds = outputs.max(dim=-1)
        # if i % 1 == 0:
        #     print('expected output:', gold_outputs[-1:])
        #     print('preds', preds[-1:])

        if i == 1000:
            print('final loss %.4f' % loss)
            print('expected output:', gold_outputs[-1:])
            print("final output:", preds[-1:])
            final_pred_seq.append(preds[-1:])

print(final_pred_seq)