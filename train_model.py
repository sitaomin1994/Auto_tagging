import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq_length) to (seq_length x batch)


def eval(dataloader, model, args, model_sign):
    model.eval()
    corrects, avg_loss = 0, 0
    for idx, batch in enumerate(dataloader, 0):
        # get the feature and target tensor
        feature, target, length = batch['feature'], batch['label'], batch['length']

        if args.cuda:
            feature, target, length = feature.cuda(), target.cuda(), length.cuda()

        feature = torch.tensor(feature).to(torch.int64)
        target = torch.tensor(target).to(torch.int64)
        length = torch.tensor(length).to(torch.int64)
        target = torch.max(target, 1)[1]

       # print(length.cpu().numpy())

        if model_sign == 'GRU':
            feature, target, length = sort_batch(feature, target, length)

        if model_sign == "GRU":
            output = model(feature, length.cpu().numpy())
        else:
            output = model(feature)
        # logit = logit.squeeze(1)
        _, predicted = torch.max(output.data, 1)
        loss = F.nll_loss(output, target, size_average=False)

        #         print(predicted.size())
        #         print(target.size())

        avg_loss += loss.item()
        corrects += torch.sum(predicted == target.data)

    size = len(dataloader.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    # print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return avg_loss, accuracy


def train(dataloader, model, args, model_sign):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    running_loss = 0.0
    corrects = 0

    for i, batch in enumerate(dataloader, 0):
        # get the feature and target tensor
        feature, target, length = batch['feature'], batch['label'], batch['length']

        if args.cuda:
            feature, target, length = feature.cuda(), target.cuda(), length.cuda()

        feature = torch.tensor(feature).to(torch.int64)
        target = torch.tensor(target).to(torch.int64)
        length = torch.tensor(length).to(torch.int64)
        target = torch.max(target, 1)[1]

        # zero the parameter gradients
        optimizer.zero_grad()

        if model_sign == 'GRU':
            feature, target, length = sort_batch(feature, target, length)
            #print(length.cpu().numpy())
        # output : model(input)
        if model_sign == "GRU":
            output = model(feature, length.cpu().numpy())
        else:
            output = model(feature)
        _, predicted = torch.max(output.data, 1)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        steps += 1
        running_loss += loss.item()
        corrects += torch.sum(predicted == target.data)

    size = len(dataloader.dataset)
    running_loss /= size
    accuracy = 100.0 * corrects / size

    # print('Epoch[{}] - loss: {:.6f}'.format(epoch, running_loss/i))

    return running_loss, accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    #print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    # return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0] + 1]

# def save(model, save_dir, save_prefix, steps):
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)
#     save_prefix = os.path.join(save_dir, save_prefix)
#     save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
#     torch.save(model.state_dict(), save_path)

def print_evaluation_scores(y_test, predicted):
    print('Accuracy:', accuracy_score(y_test, predicted))
    print('F1-score macro:', f1_score(y_test, predicted, average='macro'))
    print('F1-score micro:', f1_score(y_test, predicted, average='micro'))
    print('F1-score weighted:', f1_score(y_test, predicted, average='weighted'))
    print('Hamming_loss:', hamming_loss(y_test, predicted))