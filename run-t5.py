#!pip install sentence_transformers
#!pip install sacrebleu
#!pip install --upgrade bleu
#!pip install rouge
#!pip install datasets
#!pip install rouge_score
#%tensorflow_version 1.x (not required any more)
#!pip install -q gpt-2-simple

import argparse
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
import os
from transformers import Trainer, pipeline, set_seed, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
nltk.download('punkt')
import csv
from datasets import load_dataset, load_metric
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='t5', choices=['random', 'sbert', 'bart', 't5', 'gpt', 'gpt2', 'seq2seqlstm'])
    parser.add_argument('-s', '--setting', default='controlled', choices=['unconstrained', 'controlled', 'predict'])
    parser.add_argument('--train', default='data/wholetrain.csv') #default is for bart/t5; data format will be different for GPT
    parser.add_argument('--dev', default='data/wholedev.csv')
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--max_length', type=int, default=80)
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    return args

#Retrieval
def run_random():
    train = pd.read_csv(train_path)
    original_text = train['original_with_label'].tolist()
    hyp = []
    for i in range(835):
        hyp.append(random.choice(original_text))
    with open(os.path.join(path,'random.txt'), 'w') as f:
        for item in hyp:
            f.write("%s\n" % item)


#T5
def run_t5_unconstrained(args): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    model_checkpoint = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["original_text"]]
        # print(inputs)
        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["reframed_text"]) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
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
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

        # print(result2)
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
    print("\n****************************************\n")
    print(trainer.evaluate(tokenized_test_datasets["train"]))
    # print(trainer.evaluate(eval_dataset=tokenized_test_datasets["train"]))
    # trainer.predict(  test_dataset=tokenized_test_datasets["train"])
    # save model
    trainer.save_model("output/reframer") #TODO

    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("output/reframer", max_length=args.max_length, truncation=True)
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_unconstrained.txt'), 'w') as f:
        for i in range (len(texts)):
            f.write(texts[i] + "\t" + reframed_phrases[i] + "\n")
        # for item in reframed_phrases:
        #     f.write("%s\n" % item)
        # f.write(str(computer))

def run_t5_controlled(args): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    model_checkpoint = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "
    # prefix = ""

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["original_with_label"]]
        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            print(examples["reframed_text"][0])
            labels = tokenizer(examples["reframed_text"], max_length=args.max_length, truncation=True) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
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
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

        # print(result2)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    print(tokenized_dev_datasets["train"][0])
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
    print("\n****************************************\n")
    print(trainer.evaluate(eval_dataset=tokenized_test_datasets["train"]))
    # save model
    trainer.save_model("output/t5-control-summary") #TODO
    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained("output/t5-control-summary").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("output/t5-control-summary")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer, max_length = 50, device = 0)

    test = pd.read_csv(test_path)
    texts = test['original_with_label'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_controlled.txt'), 'w') as f:
        for i in range (len(texts)):
            f.write(texts[i] + "\t" + reframed_phrases[i] + "\n")
        # for item in reframed_phrases:
        #     f.write("%s\n" % item)

def run_t5_predict(args): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    model_checkpoint = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["original_text"]]
        model_inputs = tokenizer(inputs) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["strategy_reframe"]) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
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
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

        # print(result2)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    trainer = Seq2SeqTrainer(
        model = model,
        args = args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model("output/reframer") #TODO

    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer, max_length = 50)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_predict.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)



def main():
    #run models
    if args.model=='random':
        run_random()
    # elif args.model=='sbert':
    #     run_sbert()
    # elif args.model=='bart' and args.setting=='unconstrained':
    #     run_bart_unconstrained()
    # elif args.model=='bart' and args.setting=='controlled':
    #     run_bart_controlled()
    # elif args.model=='bart' and args.setting=='predict':
    #     run_bart_predict()
    elif args.model=='t5' and args.setting=='unconstrained':
        run_t5_unconstrained(args)
    elif args.model=='t5' and args.setting=='controlled':
        run_t5_controlled(args)
    elif args.model=='t5' and args.setting=='predict':
        run_t5_predict(args)
    # elif args.model=='gpt' and args.setting=='unconstrained':
    #     run_gpt_unconstrained()
    # elif args.model=='gpt2' and args.setting=='unconstrained':
    #     run_gpt2_unconstrained()
    # elif args.model=='seq2seqlstm' and args.setting=='unconstrained':
    #     run_seq2seqlstm_unconstrained()

# if __name__=='__main__':
args = parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
model = args.model

#load datasets
if model in ['random', 'sbert', 'bart', 't5']:
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