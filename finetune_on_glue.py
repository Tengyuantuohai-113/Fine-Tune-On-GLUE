import transformers
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import numpy as np
import torch
import random
import os
import argparse
from pynvml import *




def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True




def args():
    parser = argparse.ArgumentParser(description='An Implementation Method for Fine Tune Some Model Like BERT.')
    parser.add_argument("--dataset",default="qnli",type=str,
                          help='any dataset in GLUE')              
    parser.add_argument("--model",default="bert-base-cased",type=str,
                          help='model you want to finetune')     
    parser.add_argument("--batch_size",default=16,type=int
                          help='batch size')              
    parser.add_argument("--max_length",default=512,type=int,
                          help='max_length of sentences with padding')
    parser.add_argument("--lr",default=5e-5,type=float,
                          help='learning rate of training')
    parser.add_argument("--weight_decay",default=1e-4,type=float,
                          help='weight decay of training')
    parser.add_argument("--epochs",default=10,type=int,
                          help='epoch nums')
    parser.add_argument("--output_dir",default="./finetuned_model",type=str,
                          help='location of finetuned model storage')
    parser.add_argument("--seed",default=42,type=int,
                          help='for reproducing the results')

    args = parser.parse_args()
    return args





def fine_tune(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    def print_gpu_utilization():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")


    def print_summary(result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print_gpu_utilization()

    print_gpu_utilization()



    # pre set
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[args.dataset]

    

    # make sure parameter
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    actual_task = "mnli" if args.dataset == "mnli-mm" else args.dataset
    num_labels = 3 if args.dataset.startswith("mnli") else 1 if args.dataset=="stsb" else 2
    validation_key = "validation_mismatched" if args.dataset == "mnli-mm" else "validation_matched" if args.dataset == "mnli" else "validation"
    metric_name = "pearson" if args.dataset == "stsb" \
        else "matthews_correlation" if args.dataset == "cola"\
              else "f1" if args.dataset == "mrpc" \
                else "f1" if args.dataset == "qqp" \
                    else "accuracy"



    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)



    # dataset
    dataset = load_dataset("glue", actual_task)
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding='max_length', truncation=True, return_tensors="pt", max_length=args.max_length)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True, return_tensors="pt", max_length=args.max_length)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)     



    # bulid mdoel
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)



    # metric
    metric = evaluate.load('glue', actual_task)



    # train & eval
    training_args = TrainingArguments(
    output_dir=f"{args.output_dir}/{args.model}_{args.dataset}",
    evaluation_strategy='epoch',
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    load_best_model_at_end=True,
    )


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=-1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)


    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


    trainer.train()





if __name__ == "__main__":
    _args = args()
    fine_tune(_args)
