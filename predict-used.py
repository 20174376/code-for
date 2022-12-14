from ast import arg
import os
import pandas as pd
import numpy as np
import argparse
from transformers import Trainer, pipeline, set_seed, AutoTokenizer, \
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generatetype', default='pipeline', choices=['pipeline', 'vanillagenerate'])
    parser.add_argument('-m', '--model', default='bart', choices=['bart', 't5'])
    parser.add_argument("--use_prompt", type=bool, default=False, help="Whether to use prompt Only for t5.",)
    parser.add_argument('--mode', default='1to1', choices=['1to1', '1tomany'], help="generate 1to1 or 1tomany file ")
    parser.add_argument('-s', '--setting', default='control', choices=['uncontrol', 'control', 'predict'])
    parser.add_argument('--numbercount', default=5, type=int, help="In 1 to many, the number of many")
    parser.add_argument('--test_path', default='data/wholetest.csv')
    parser.add_argument('--max_length', type=int, default=80)
    parser.add_argument('--model_path', type=str, default="output-bart/bart-control-lossbice-three-ep10")
    parser.add_argument('--output_path', type=str, default='output-bart/bart-control-lossbice-three-ep10.txt')
    parser.add_argument("--device", default=None)
    # parser.add_argument("--model_path", default="output/reframer", type=str, help="the model path you have trained")
    # parser.add_argument("--flag", default=True, help="whether to use self generated test file")
    args = parser.parse_args()
    return args
args = parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
def Pipeline1to1(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    test = pd.read_csv(args.test_path)
    # test = load_dataset('csv', data_files=test_path)
    # test = test.dropna()
    if args.setting == "uncontrol" or args.setting == "predict":
        original = test['original_text'].tolist()
    else:
        original = test['original_with_label'].tolist()

    # print(texts[0])
    # inputs = ['summarize: ' + text for text in test['original_text']]
    inputs = []
    if args.model == "t5":
        if args.use_prompt == True: 
            for ori in original:
                input  = "Here is some text: " + ori + " Here is a rewrite of the text, which is more positive: "
                inputs.append(input)
        else:
            for text in original:
                inputs.append("summarize: " + text)
    else:
        if args.use_prompt == True: 
            for ori in original:
                input  = "Here is some text: " + ori + " Here is a rewrite of the text, which is more positive: "
                inputs.append(input)
        else:
            inputs = original[:]
    
    print("sample:\n")
    print("\t" + inputs[0])
    # print(inputs)
    print("\n***************\n")

    fw = open(args.output_path, "w")
    for i in range(len(inputs)):
        reframer = pipeline ("summarization", model=model, 
                            num_return_sequences = 1,
                            # num_beams=6,    
                            # do_sample=True, 
                            # top_k=60,
                            # top_p=0.90,
                            # typical_p = 0.20,
                            # shabi = 1,
                            # temperature = 1.2,
                            device = 0, tokenizer=tokenizer,max_length = 50)
        outputs = reframer(inputs[i])
        outputs_dec = [output['summary_text'] for output in outputs]
        
        
        # outputs_dec = [tokenizer.decode(ids, skip_special_tokens=True)for ids in outputs]
        ## 这里用原始的original方便测试
        fw.write(original[i] + "\t" + outputs_dec[0]+ "\n")
    print("complete")
    fw.close()


def Pipeline1tomany(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    test = pd.read_csv(args.test_path)
    # test = load_dataset('csv', data_files=test_path)
    # test = test.dropna()
    if args.setting == "uncontrol" or args.setting == "predict":
        original = test['original_text'].tolist()
    else:
        original = test['original_with_label'].tolist()


    # print(texts[0])
    # inputs = ['summarize: ' + text for text in test['original_text']]
    inputs = []
    if args.model == "t5": 
        if args.use_prompt == True: 
            for ori in original:
                input  = "Here is some text: " + ori + " Here is a rewrite of the text, which is more positive: "
                inputs.append(input)
        else:
            for text in original:
                inputs.append("summarize: " + text)
    else:
        if args.use_prompt == True: 
            for ori in original:
                input  = "Here is some text: " + ori + " Here is a rewrite of the text, which is more positive: "
                inputs.append(input)
        else:
            inputs = original[:]
    print("sample:\n")
    print("\t" + inputs[0])
    # print(inputs)
    print("\n***************\n")

    fw = open(args.output_path, "w")
    count = 0
    for input in inputs:
        # input = input.to(device)
        reframer = pipeline ("summarization", model=model, num_return_sequences = args.numbercount,
                                device = 0, tokenizer=tokenizer, 
                                num_beams=5, 
                                do_sample=True, 
                                top_k=50,
                                # top_p=0.9,
                                # typical_p = 0.20,
                                max_length = 50)
        outputs = reframer(input)
        outputs_dec = [output['summary_text'] for output in outputs]
        for i in range (args.numbercount):
            fw.write(original[count] + "\t" + outputs_dec[i]+ "\n")
        count = count + 1
    print("complete")
    fw.close()

def VanillaGenerate1to1(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    test = pd.read_csv(args.test_path)
    # test = load_dataset('csv', data_files=test_path)
    # test = test.dropna()
    if args.setting == "uncontrol" or args.setting == "predict":
        original = test['original_text'].tolist()
    else:
        original = test['original_with_label'].tolist()


    # print(texts[0])
    # inputs = ['summarize: ' + text for text in test['original_text']]
    inputs = []
    if args.model == "t5":
        if args.use_prompt == True: 
            for ori in original:
                # input  = "Here is some text: " + ori + " Here is a rewrite of the text, which is more positive: "
                input = "Here is a text: <" + ori + "> Here is a rewrite of the text, which is more positive: <" 
                inputs.append(input)
        else:
            for text in original:
                inputs.append("summarize: " + text)
    else:
        if args.use_prompt == True: 
            for ori in original:
                input  = "Here is some text: " + ori + " Here is a rewrite of the text, which is more positive: "
                inputs.append(input)
        else:
            inputs = original[:]
    print("sample:\n")
    print("\t" + inputs[0])
    # print(inputs)
    print("\n***************\n")

    fw = open(args.output_path, "w")
    
    
    for i in range(len(inputs)):
        ##这里是用了提示
        # input  = "Here is some text: " + original[i] + " Here is a rewrite of the text, which is more positive:"
        encode_text = tokenizer(inputs[i], max_length=args.max_length, return_tensors="pt",padding=True, truncation=True)
        encode_text = encode_text.to(args.device)
        outputs = model.generate(encode_text['input_ids'],
                                attention_mask=encode_text['attention_mask'],
                                max_length = 50,
                                # min_length = 20,
                                num_beams = 5,
                                # do_sample = True,
                                # top_k = 40,
                                # top_p = 0.9,
                                # typical_p = 0.5,
                                # temperature = 1.1,
                                repetition_penalty = 2.0,
                                num_return_sequences = 1)
        outputs_dec = [tokenizer.decode(ids, skip_special_tokens=True)for ids in outputs]
        # print(1)
        fw.write(original[i] + "\t" + outputs_dec[0]+ "\n")
    
    print("complete")
    fw.close()



def VanillaGenerate1tomany(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    test = pd.read_csv(args.test_path)
    # test = load_dataset('csv', data_files=test_path)
    # test = test.dropna()
    
    if args.setting == "uncontrol" or args.setting == "predict":
        original = test['original_text'].tolist()
    else:
        original = test['original_with_label'].tolist()


    # print(texts[0])
    # inputs = ['summarize: ' + text for text in test['original_text']]
    inputs = []
    if args.model == "t5":
        if args.use_prompt == True: 
            for ori in original:
                input  = "Here is some text: " + ori + " Here is a rewrite of the text, which is more positive: "
                inputs.append(input)
        else:
            for text in original:
                inputs.append("summarize: " + text)
    else:
        if args.use_prompt == True: 
            for ori in original:
                input  = "Here is some text: " + ori + " Here is a rewrite of the text, which is more positive: "
                inputs.append(input)
        else:
            inputs = original[:]
    print("sample:\n")
    print("\t" + inputs[0])
    # print(inputs)
    print("\n***************\n")

    fw = open(args.output_path, "w")
    count = 0 #计数
    for input in inputs:
        encode_text = tokenizer(input, max_length=args.max_length, return_tensors="pt")
        encode_text = encode_text.to(args.device)
        outputs = model.generate(encode_text['input_ids'],
                                attention_mask=encode_text['attention_mask'],
                                max_length = 50,
                                # min_length = 20,
                                num_beams = 8,
                                # do_sample = True,
                                # top_k = 30,
                                # top_p = 0.99,
                                # typical_p = 0.95,
                                # diversity_penalty = 0.2,
                                # num_beam_groups = 4, ##可以防止出现UNK
                                # temperature = 1.1,
                                remove_invalid_values = True,
                                num_return_sequences = args.numbercount)
        # outputs_dec = [tokenizer.decode(ids, skip_special_tokens=True)for ids in outputs]
        outputs_dec = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range (args.numbercount):
            fw.write(original[count] + "\t" + outputs_dec[i]+ "\n")
        count = count + 1
            # fw.write("%s\n" % outputs_dec[0])
        # count = count + 1
        # print(count)
    fw.close()
    print("complete")

if args.generatetype == "pipeline" and args.mode == "1to1":
    Pipeline1to1(args)
elif args.generatetype == "pipeline" and args.mode == "1tomany":
    Pipeline1tomany(args)
elif args.generatetype == "vanillagenerate" and args.mode == "1to1":
    VanillaGenerate1to1(args)
elif args.generatetype == "vanillagenerate" and args.mode == "1tomany":
    VanillaGenerate1tomany(args)

