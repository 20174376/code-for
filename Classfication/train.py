# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import json
from utils import get_time_dif, gettoken
from transformers import AdamW, get_linear_schedule_with_warmup


def train(args, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    param_optimizer = list(model.named_parameters()) #Bert模型每一层的深度学习模型
    param_optimizer = [n for n in param_optimizer]
    #print('param_optimizer:', param_optimizer)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #这个是一个训练时候的策略，分层训练。当使用bert+下游模型训练的时候，因为bert已经在大规模数据集上预训练好了，
    # 所以使用较低的学习率（lr）来训练，下游的模型就使用较高的学习率（u_lr）训练。
    optimizer_parameters  = [
        {'params': [p for n, p in param_optimizer if 'bert' in n and not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if 'bert' in n and any(nd in n for nd in no_decay)],
        'weight_decay': 0.0, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if 'bert' not in n and not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay, 'lr': args.u_lr},
        {'params': [p for n, p in param_optimizer if 'bert' not in n and any(nd in n for nd in no_decay)],
        'weight_decay': 0.0, 'lr': args.u_lr}
    ]
    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_iter) * args.epochs)
    # model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = 1e12
    dev_best_f1 = 0
    model.train()
    for epoch in range(args.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
        for i, batches in enumerate(train_iter):
            model.zero_grad()
            sent, labels = batches
            input_ids, attention_mask, type_ids, position_ids = gettoken(args, sent)
            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(args.device), attention_mask.to(args.device), type_ids.to(args.device), labels.to(args.device)
            position_ids = position_ids.to(args.device)
            pmi = model(input_ids, attention_mask, type_ids, position_ids)
            loss = F.binary_cross_entropy(pmi, labels.float(), reduction='sum')
            # focal_loss =  BCEFocalLoss()
            # loss = focal_loss(pmi, labels.float())
            # balace_loss = Balanced_CE_loss()
            # loss = balace_loss(pmi, labels.float())
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_batch += 1
            if total_batch % 100 == 0:
                # print(list(model.named_parameters()))
                time_dif = get_time_dif(start_time)
                print("test:")
                f1, _, dev_loss, predict, ground, sents = evaluate(args, model, dev_iter, test=False)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Time: {2}'
                print(msg.format(total_batch, loss.item(), time_dif))
                print("loss", total_batch, loss.item(), "dev_loss:", dev_loss)
                # if dev_loss < dev_best_loss:
                #     print("save", dev_loss)
                #     torch.save(model.state_dict(), args.save_path + "model.ckpt")
                #     dev_best_loss = dev_loss
                if f1 > dev_best_f1:
                    print("save", f1)
                    torch.save(model, args.save_path + args.strategy_type + args.prefix + args.model_type + ".ckpt")
                    dev_best_f1 = f1
                model.train()

    test(args, model, test_iter)


def evaluate(args, model, data_iter, test=True):
    model.eval()
    loss_total = 0
    predicts, sents, grounds, all_bires = [], [], [], []
    with torch.no_grad():
        for i, batches in enumerate(data_iter):
            sent, labels = batches
            input_ids, attention_mask, type_ids, position_ids = gettoken(args,sent)
            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(args.device), attention_mask.to(args.device), type_ids.to(args.device), labels.to(
                    args.device)
            position_ids = position_ids.to(args.device)
            pmi = model(input_ids, attention_mask, type_ids, position_ids)
            loss = F.binary_cross_entropy(pmi, labels.float(), reduction='sum')
            # focal_loss =  BCEFocalLoss()
            # loss = focal_loss(pmi, labels.float())
            # balace_loss = Balanced_CE_loss()
            # loss = balace_loss(pmi, labels.float())
            loss_total += loss.item()
            bires = torch.where(pmi > 0.5, torch.tensor([1]).to(args.device), torch.tensor([0]).to(args.device))
            for b, g, p, s in zip(bires, labels, pmi, sent):
                all_bires.append(b.item())
                predicts.append(p.item())
                grounds.append(g.item())
                sents.append(s)
    print("test set size:", len(grounds))
    accuracy = metrics.accuracy_score(grounds, all_bires)
    p = metrics.precision_score(grounds, all_bires, zero_division=0)
    r = metrics.recall_score(grounds, all_bires, zero_division=0)
    f1 = metrics.f1_score(grounds, all_bires, zero_division=0)
    print("f1:{},p:{},r,{}, accuracy:{}".format(f1, p, r, accuracy))
    return f1, pmi, loss_total / len(data_iter), predicts, grounds, sents


def predict(args, model, data_iter):
    model.eval()
    predicts, sents, grounds, all_bires = [], [], [], []
    with torch.no_grad():
        for i, batches in enumerate(data_iter):
            sent, labels = batches
            input_ids, attention_mask, type_ids, position_ids = gettoken(args,sent)
            input_ids, attention_mask, type_ids = \
                input_ids.to(args.device), attention_mask.to(args.device), type_ids.to(args.device)
            position_ids = position_ids.to(args.device)
            pmi = model(input_ids, attention_mask, type_ids, position_ids)
            bires = torch.where(pmi > 0.5, torch.tensor([1]).to(args.device), torch.tensor([0]).to(args.device))
            for b, g, p, s in zip(bires, labels, pmi, sent):
                all_bires.append(b.item())
                predicts.append(p.item())
                grounds.append(g.item())
                sents.append(s)
    accuracy = metrics.accuracy_score(grounds, all_bires)
    p = metrics.precision_score(grounds, all_bires, zero_division=0)
    r = metrics.recall_score(grounds, all_bires, zero_division=0)
    f1 = metrics.f1_score(grounds, all_bires, zero_division=0)
    print("f1:{},p:{},r,{}, accuracy:{}".format(f1, p, r, accuracy))
    with open(args.save_path + args.strategy_type + args.prefix + args.model_type + ".txt", "w") as f:
        for b in all_bires:
            f.write("result: " + str(b)+"\n")
        f.writelines("f1:{},p:{},r:{}, accuracy:{}".format(f1, p, r, accuracy))
    #         for b, t in zip(bires, triple_id):
    #             # predicts.append({"salience": b.item(), "triple_id": t})
    #             predicts.append({"triple_id": t, "salience": b.item()})
    # with open(args.save_path + "sen_erine_Last3andPool_bs32_len32_K"+str(index)+".jsonl", "w") as f:
    #     for t in predicts:
    #         f.write(json.dumps(t, ensure_ascii=False)+"\n")


def test(args, model, test_iter):
    # test
    model = torch.load(args.save_path+ args.strategy_type + args.prefix + args.model_type + ".ckpt")
    # model.load_state_dict(torch.load(args.save_path+ args.strategy_type + args.model_type + ".ckpt"))
    model.eval()
    start_time = time.time()
    predict(args, model, test_iter,)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)