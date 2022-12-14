# coding: UTF-8
from ast import arg
from re import I
import time, os
from jinja2 import pass_environment
import numpy as np
from train import train, test
import random
from models import *
import argparse
from utils import build_dataset, build_iterator, get_time_dif, load_dataset, gettoken
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
from sklearn import metrics, model_selection
import pandas as pd
import csv

os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train_entry(args):
    start_time = time.time()
    print("Loading data...")
    train_data = load_dataset(args.train_path, args)
    dev_data = load_dataset(args.dev_path, args)
    test_data = load_dataset(args.test_path, args)
    random.shuffle(train_data)
    # random.shuffle(dev_data)
    # random.shuffle(test_data)
    train_iter = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.batch_size,
        # num_workers=args.num_workers,
        drop_last=False)
    dev_iter = DataLoader(dev_data, shuffle=True, batch_size=args.batch_size,
                           drop_last=False)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=args.batch_size,
                            drop_last=False)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    # model = Bert_Last3andPool(args).to(args.device)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = eval(args.model_type)(args).to(args.device)
    model = nn.DataParallel(model)
    train(args, model, train_iter, dev_iter, test_iter)


def test_entry(args):
    test_data = load_dataset(args.test_path, args)
    # model = eval(args.model_type)(args).to(args.device)
    # model.load_state_dict(torch.load(args.save_path + args.model_type + ".ckpt"))
    # torch.save(model, args.save_path+ args.model_type + "_all.ckpt")
    model = torch.load(args.save_path + args.strategy_type + args.prefix + args.QAtype + args.model_type + ".ckpt")
    # model = nn.DataParallel(model)
    model.eval()
    predicts, sents, grounds, all_bires = [], [], [], []
    loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
    
    for i, batches in enumerate(loader):
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
    with open(args.save_path + args.strategy_type + args.prefix + args.model_type + ".txt", "w") as f:
        for t in predicts:
            f.write(json.dumps(t, ensure_ascii=False)+"\n")
        f.writelines("f1:{},p:{},r:{}, accuracy:{}".format(f1, p, r, accuracy))
    print("f1:{},p:{},r:{}, accuracy:{}".format(f1, p, r, accuracy))


def score_sentence (args):
    strategy_list = ["Growth", "Neutralizing", "Optimism", "Self_affirmation", "Impermanence", "Thankfulness"]
    args.strategy_type = strategy_list[0]
    score_data_gro = load_dataset(args.score_path, args)
    
    args.strategy_type = strategy_list[1]
    score_data_neu = load_dataset(args.score_path, args)
    
    args.strategy_type = strategy_list[2]
    score_data_opt = load_dataset(args.score_path, args)
    
    args.strategy_type = strategy_list[3]
    score_data_sel = load_dataset(args.score_path, args)
    
    args.strategy_type = strategy_list[4]
    score_data_imp = load_dataset(args.score_path, args)
    
    args.strategy_type = strategy_list[5]
    score_data_tha = load_dataset(args.score_path, args)
    
    model_Growth = torch.load(args.save_path+ "bestGrowth" + args.prefix + args.model_type + ".ckpt")
    model_Neutralizing = torch.load(args.save_path+ "bestNeutralizing" + args.prefix + args.model_type + ".ckpt")
    model_Optimism = torch.load(args.save_path+ "bestOptimism" + args.prefix + args.model_type + ".ckpt")
    model_Self_affirmation = torch.load(args.save_path+ "bestSelf_affirmation" + args.prefix + args.model_type + ".ckpt")
    # model_Self_affirmation = torch.load("Classfication/output/bestSelf_affirmationQAMBartAugVanillaRoBert.ckpt")
    
    model_Impermanence = torch.load(args.save_path+ "bestImpermanence" + args.prefix + args.model_type + ".ckpt")
    model_Thankfulness = torch.load(args.save_path+ "bestThankfulness" + args.prefix + args.model_type + ".ckpt")
    
    # model_Growth = nn.DataParallel(model_Growth)
    model_Growth.eval()
    # model_Neutralizing = nn.DataParallel(model_Neutralizing)
    model_Neutralizing.eval()
    # model_Optimism = nn.DataParallel(model_Optimism)
    model_Optimism.eval()
    ## Self需要DP
    # model_Self_affirmation = nn.DataParallel(model_Self_affirmation)
    model_Self_affirmation.eval()
    # model_Impermanence = nn.DataParallel(model_Impermanence)
    model_Impermanence.eval()
    # model_Thankfulness = nn.DataParallel(model_Thankfulness)
    model_Thankfulness.eval()
    
    predicts, sents, grounds, all_bires = [], [], [], []
    loader_gro = DataLoader(score_data_gro, shuffle=False, batch_size=args.test_batch)
    loader_neu = DataLoader(score_data_neu, shuffle=False, batch_size=args.test_batch)
    loader_opt = DataLoader(score_data_opt, shuffle=False, batch_size=args.test_batch)
    loader_sel = DataLoader(score_data_sel, shuffle=False, batch_size=args.test_batch)
    loader_imp = DataLoader(score_data_imp, shuffle=False, batch_size=args.test_batch)
    loader_tha = DataLoader(score_data_tha, shuffle=False, batch_size=args.test_batch)
    
    f = open (args.save_path + args.score_output, "w+")
    file = pd.read_csv("/nfs/users/jiashutong/vscodefile/Refram/positive-frames-main/data/wholetest.csv")
    strategy = file['strategy'].tolist()
    
    for i, (gro, neu, opt, sel, imp, tha) in enumerate(zip(loader_gro, loader_neu, loader_opt, loader_sel, loader_imp, loader_tha)):
        sent_gro, labels_gro = gro
        # if (i==28):
        #     print(1)
        input_ids_gro, attention_mask_gro, type_ids_gro, position_ids_gro = gettoken(args,sent_gro)
        input_ids_gro, attention_mask_gro, type_ids_gro = \
            input_ids_gro.to(args.device), attention_mask_gro.to(args.device), type_ids_gro.to(args.device)
        position_ids_gro = position_ids_gro.to(args.device)
                
        sent_neu, labels_neu = neu
        input_ids_neu, attention_mask_neu, type_ids_neu, position_ids_neu = gettoken(args,sent_neu)
        input_ids_neu, attention_mask_neu, type_ids_neu = \
            input_ids_neu.to(args.device), attention_mask_neu.to(args.device), type_ids_neu.to(args.device)
        position_ids_neu = position_ids_neu.to(args.device)
        
        sent_opt, labels_opt = opt
        input_ids_opt, attention_mask_opt, type_ids_opt, position_ids_opt = gettoken(args,sent_opt)
        input_ids_opt, attention_mask_opt, type_ids_opt = \
            input_ids_opt.to(args.device), attention_mask_opt.to(args.device), type_ids_opt.to(args.device)
        position_ids_opt = position_ids_opt.to(args.device)
        
        sent_sel, labels_sel = sel
        input_ids_sel, attention_mask_sel, type_ids_sel, position_ids_sel = gettoken(args,sent_sel)
        input_ids_sel, attention_mask_sel, type_ids_sel = \
            input_ids_sel.to(args.device), attention_mask_sel.to(args.device), type_ids_sel.to(args.device)
        position_ids_sel = position_ids_sel.to(args.device)
        
        sent_imp, labels_imp = imp
        input_ids_imp, attention_mask_imp, type_ids_imp, position_ids_imp = gettoken(args,sent_imp)
        input_ids_imp, attention_mask_imp, type_ids_imp = \
            input_ids_imp.to(args.device), attention_mask_imp.to(args.device), type_ids_imp.to(args.device)
        position_ids_imp = position_ids_imp.to(args.device)
        
        sent_tha, labels_tha = tha
        input_ids_tha, attention_mask_tha, type_ids_tha, position_ids_tha = gettoken(args,sent_tha)
        input_ids_tha, attention_mask_tha, type_ids_tha = \
            input_ids_tha.to(args.device), attention_mask_tha.to(args.device), type_ids_tha.to(args.device)
        position_ids_tha = position_ids_tha.to(args.device)
        
        pmi = 0
        flag = 0
        # print(strategy[i//args.numbercount])
        if "growth" in strategy[i//args.numbercount]:          
            pmi += model_Growth(input_ids_gro, attention_mask_gro, type_ids_gro, position_ids_gro)
            flag = 1
        if "neutralizing" in strategy[i//args.numbercount]:
            flag = 1
            pmi += model_Neutralizing(input_ids_neu, attention_mask_neu, type_ids_neu, position_ids_neu)
        if "optimism" in strategy[i//args.numbercount]:
            flag = 1
            pmi += model_Optimism(input_ids_opt, attention_mask_opt, type_ids_opt, position_ids_opt)
        if "self_affirmation" in strategy[i//args.numbercount]:
            flag = 1
            pmi += model_Self_affirmation(input_ids_sel, attention_mask_sel,type_ids_sel, position_ids_sel)
        if "impermanence" in strategy[i//args.numbercount]:
            flag = 1
            pmi += model_Impermanence(input_ids_imp, attention_mask_imp, type_ids_imp, position_ids_imp)
        if "thankfulness" in strategy[i//args.numbercount]:
            flag = 1
            pmi += model_Thankfulness(input_ids_tha, attention_mask_tha, type_ids_tha, position_ids_tha)
        if flag == 0:
            print(strategy[i//args.numbercount])
            time.sleep(100)
        # print(i)
        f.writelines(str(pmi.item()) + "\n")
        print(i)
    f.close()

parser = argparse.ArgumentParser(description='Judege positive reframe')
parser.add_argument("--do_train", type=bool, default=False, help="Whether to run training.",)
parser.add_argument("--do_score", type=bool, default=True, help="Whether to run scoring",)


parser.add_argument("--test_batch", default=1, type=int, help="Test every X updates steps.")
# parser.add_argument("--test_batch", default=8, type=int, help="Test every X updates steps.")

parser.add_argument("--train_path", default="data/class/", type=str, help="The train data directory.")
parser.add_argument("--dev_path", default="data/class/", type=str, help="The dev data directory.")
parser.add_argument("--test_path", default="data/class/", type=str, help="The test data directory.")

parser.add_argument("--score_path", default="output-t5/t5-control-loss-ep10-tp20-95.txt", type=str, help="The score data directory.")
parser.add_argument("--score_output", default="class_score-t5-loss-ep10-tp20-95-bert.txt", type=str, help="The score data directory.")
parser.add_argument('--numbercount', default=2, type=int, help="In 1 to many, the number of many")

# parser.add_argument("--model_path", default="/nfs/users/jiashutong/PLM/Roberta-En-Base", type=str, help="The directory of pretrained models")
parser.add_argument("--model_path", default="/nfs/users/jiashutong/PLM/bert-base-uncased", type=str, help="The directory of pretrained models")

parser.add_argument("--model_type", type=str, default="VanillaBert", help="the model you want to train")
parser.add_argument("--save_path", default='Classfication/output/', type=str, help="The path of result data and models to be saved.")
parser.add_argument("--device", default=None)
# parser.add_argument("--tokenizer", default=None)
# models param
parser.add_argument("--max_length", default=110, type=int, help="the max length of sentence.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training.")
#BERT 5e-5， 1e-5
parser.add_argument("--learning_rate", default=11e-6, type=float, help="The initial learning rate for Adam.")
parser.add_argument('--u_lr', type=float, default=18e-5) #u_lr是下层学习率  原定1e-4
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--dropout", default=0.1, type=float, help="Drop out rate")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('--hidden_size', type=int, default=768,  help="random seed for initialization")


parser.add_argument("--prefix", type=str, default="QA", help="Task conversion prefix")
parser.add_argument("--QAtype", type=str, default="", help="Prefix of question type")
parser.add_argument("--strategy_type", type=str, default="Thankfulness", choices=["Growth", "Neutralizing",
                                                "Optimism", "Self_affirmation", "Impermanence", "Thankfulness"])

args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args.train_path = args.train_path + "Train"  + args.strategy_type + ".txt"
# args.train_path = "data/class/TrainAugSelf_affirmation.txt"
args.dev_path = args.dev_path + "Dev" + args.strategy_type + ".txt"
args.test_path = args.test_path + "Test" + args.strategy_type + ".txt"

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

if args.do_score:
    score_sentence(args)
elif not args.do_train:
    test_entry(args)
else:
    train_entry(args)