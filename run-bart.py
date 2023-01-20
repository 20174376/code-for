#!pip install sentence_transformers
#!pip install sacrebleu
#!pip install --upgrade bleu
#!pip install rouge
#!pip install datasets
#!pip install rouge_score
#%tensorflow_version 1.x (not required any more)
#!pip install -q gpt-2-simple

import argparse
from cProfile import label
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
import os
from transformers import Trainer, pipeline, set_seed, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
# nltk.download('punkt')
import csv
from datasets import load_dataset, load_metric
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# from JudgeReframe.utils import *
# from JudgeReframe.utils import build_dataset, build_iterator, get_time_dif
# from JudgeReframe.utils import load_dataset as ld
from JudgeReframe.models import *
from JudgeReframe.utils import build_dataset, build_iterator, get_time_dif, gettoken
from JudgeReframe.utils import load_dataset as ld
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import math
from statistics import mean
from torch.cuda.amp import autocast



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='bart', choices=['bart', 't5'])
    parser.add_argument('-s', '--setting', default='unconstrained', choices=['unconstrained', 'controlled', 'predict'])
    parser.add_argument('--train', default='data/wholetrain.csv') #default is for bart/t5; data format will be different for GPT
    parser.add_argument('--dev', default='data/wholedev.csv')
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--max_length', type=int, default=80)
    parser.add_argument('--output_dir', type=str, default='output-bart/')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model_type", type=str, default="VanillaRoBert", help="the model you want to train")
    parser.add_argument("--dropout", default=0.1, type=float, help="Drop out rate")
    parser.add_argument('--hidden_size', type=int, default=1024,  help="random seed for initialization")
    parser.add_argument("--model_path", default="/nfs/users/jiashutong/PLM/Roberta-En-Large", type=str, help="The directory of pretrained models")
    args = parser.parse_args()
    return args

def judge_reframe(labels, reframe):
    train_dataset = pd.read_csv('data/wholetrain.csv')
    original_all = train_dataset["original_text"].to_list()
    reference = train_dataset["reframed_text"].to_list()
    index_list = [] ##查找labels中每个句子对应的原始句子
    for label in labels:
        if label in reference:
            index_list.append(reference.index(label))
        else:
            print("may be dev, not found!")
            return 1  ##在验证和测试中不需要该loss，后边是1减去，所以这里直接减1
    original = []
    for index in index_list:
        original.append(original_all[index])
    fw = open("judge_temple-bart.txt", "w+")
    for i in range (len(original)):
        fw.write(original[i] + "\t" + reframe[i] + "\n")
    fw.close()
    # judegemodel = eval(args.model_type)(args).to(args.device)
    # judegemodel = nn.DataParallel(judegemodel)
    # judegemodel.load_state_dict(torch.load("JudgeReframe/output/VanillaRoBertLarge-canshu.ckpt"))
    # # model.load_state_dict(torch.load("JudgeReframe/output/VanillaBert.ckpt"))
    # # model = torch.load("JudgeReframe/output/VanillaBert_all.ckpt").to(args.device)
    # judegemodel.eval()
    test_data = ld("judge_temple-bart.txt")
    loader = DataLoader(test_data, shuffle=False, batch_size=1,drop_last=False)
    predicts= []
    loss = []
    label = [1]
    label = torch.tensor(label)
    for i, batches in enumerate(loader):
        sent, labels = batches
        input_ids, attention_mask, type_ids, position_ids = gettoken(args,sent)
        input_ids, attention_mask, type_ids = \
            input_ids.to(args.device), attention_mask.to(args.device), type_ids.to(args.device)
        position_ids = position_ids.to(args.device)
        pmi = judegemodel(input_ids, attention_mask, type_ids, position_ids)
        # bires = torch.where(pmi > 0.5, torch.tensor([1]).to(args.device), torch.tensor([0]).to(args.device))
         ## 防止精度问题
        with autocast(enabled=False):
            loss.append(F.binary_cross_entropy(pmi.float(), label.float().to(args.device)))
        for p in pmi:
            predicts.append(p.item())
    # print(predicts)
    loss = torch.tensor(loss).to(args.device)
    score = mean(predicts)
    loss_sum = loss.sum()
    return loss_sum

            

#BART
def run_bart_unconstrained(args):
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("/nfs/users/jiashutong/PLM/bart-base")
    def preprocess_function(examples):
        inputs = examples["original_text"]
        # inputs = ["Here is some text: " + doc + " Here is a rewrite of the text, which is more positive:" for doc in examples["original_text"]]
        # print(inputs[0])
        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["reframed_text"], max_length=args.max_length, truncation=True) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    from transformers import BartForConditionalGeneration, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
    model = BartForConditionalGeneration.from_pretrained("/nfs/users/jiashutong/PLM/bart-base")
    batch_size = 12
    
    
    args = Seq2SeqTrainingArguments(
        "bart-summarization",
        evaluation_strategy = "steps",
        save_steps=100,
        save_strategy='steps',
        learning_rate=3e-5,
        eval_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=10,
        predict_with_generate=True,
        load_best_model_at_end = True ,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {key: value.mid.fmeasure for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        # Add bleu scores
        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    # trainer.evaluate()
    print("\n****************************************\n")
    print(trainer.evaluate(eval_dataset=tokenized_test_datasets["train"]))
    trainer.save_model("output-bart/bart-uncontrol")
    # Load trained model
    # model = AutoModelForSeq2SeqLM.from_pretrained("output-bart/bart-uncontrol")
    # tokenizer = AutoTokenizer.from_pretrained("output-bart/bart-uncontrol")
    # reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    # test = pd.read_csv(test_path)
    # texts = test['original_text'].to_list()
    # reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    # with open(os.path.join(path,'bart_unconstrained.txt'), 'w') as f:
    #     for i in range (len(texts)):
    #         f.write(texts[i] + "\t" + reframed_phrases[i] + "\n")
            
def run_bart_controlled(args):
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("/nfs/users/jiashutong/PLM/bart-base")
    def preprocess_function(examples):
        inputs = examples["original_with_label"]
        model_inputs = tokenizer(inputs, max_length=80, truncation=True) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["reframed_text"], max_length=args.max_length, truncation=True) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    from transformers import BartForConditionalGeneration, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
    model = BartForConditionalGeneration.from_pretrained("/nfs/users/jiashutong/PLM/bart-base")
    batch_size = 12
    
    class CustomTrainer(Seq2SeqTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs, return_dict=True)
            # selfmodel = model.module  ##DP时需要
            # print(len(inputs["input_ids"]))
            # input_ids = inputs.input_ids
            # tokenizer.batch_decode(**inputs, skip_special_tokens=True)
            # input_ids = inputs["input_ids"].cpu().detach().numpy().tolist()
            labels = inputs["labels"].cpu()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # original = []
            # for input in input_ids:
            #     string  = ""
            #     token_list = tokenizer.convert_ids_to_tokens(input, skip_special_tokens=True) 
            #     for token in token_list:
            #         string = string + token
            #     original.append(string)
            loss = outputs["loss"]
            
            out = model.generate(input_ids=inputs["input_ids"], 
                                     attention_mask=inputs["attention_mask"],
                                     remove_invalid_values = True)
            reframe = tokenizer.batch_decode(out, skip_special_tokens=True)
            # print(decoded_labels)
            reframe_loss = judge_reframe(decoded_labels, reframe)
            
            # print(loss)
            loss += reframe_loss
            return outputs["loss"] + reframe_loss
    
    args = Seq2SeqTrainingArguments(
        "bart-summarization",
        evaluation_strategy = "steps",
        save_steps=100,
        save_strategy='steps',
        learning_rate=3e-5,
        eval_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=10,
        predict_with_generate=True,
        load_best_model_at_end = True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {key: value.mid.fmeasure for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        # Add bleu scores
        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    trainer = CustomTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    # trainer.evaluate()
    print("\n****************************************\n")
    print(trainer.evaluate(eval_dataset=tokenized_test_datasets["train"]))
    trainer.save_model("output-bart/bart-control")
    # Load trained model
    # model = AutoModelForSeq2SeqLM.from_pretrained("output-bart/bart-control")
    # tokenizer = AutoTokenizer.from_pretrained("output-bart/bart-control")
    # reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    # test = pd.read_csv(test_path)
    # texts = test['original_with_label'].to_list()
    # reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    # with open(os.path.join(path,'bart_controlled.txt'), 'w') as f:
    #     for i in range (len(texts)):
    #         f.write(texts[i] + "\t" + reframed_phrases[i] + "\n")
    

def main():
    #run models
    if args.model=='bart' and args.setting=='unconstrained':
        run_bart_unconstrained(args)
    elif args.model=='bart' and args.setting=='controlled':
        run_bart_controlled(args)

# if __name__=='__main__':
args = parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
model = args.model

judegemodel = eval(args.model_type)(args).to(args.device)
judegemodel = nn.DataParallel(judegemodel)
# judegemodel.load_state_dict(torch.load("JudgeReframe/output/VanillaRoBertLarge-canshu.ckpt"))
judegemodel.load_state_dict(torch.load("JudgeReframe/output/bestThreeVanillaRoBertLarge.ckpt"))
# model.load_state_dict(torch.load("JudgeReframe/output/VanillaBert.ckpt"))
# model = torch.load("JudgeReframe/output/VanillaBert_all.ckpt").to(args.device)
judegemodel.eval()

#load datasets
if model in ['random', 'sbert', 'bart', 't5','ctrl']:
    train_path = args.train
    train_dataset = load_dataset('csv', data_files=train_path)
    dev_path = args.dev
    dev_dataset = load_dataset('csv', data_files=dev_path)
    test_path = args.test
    test_dataset = load_dataset('csv', data_files=test_path)
elif model in ['gpt', 'gpt2']:
    train_path = args.train
    test_path = args.test


else:
    raise Exception("Sorry, this model is currently not included.")

path=args.output_dir
main()