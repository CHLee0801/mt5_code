from datasets import Pretrain
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration
import csv
import os 
import re
import string
from rouge import Rouge
from tqdm import tqdm
from collections import Counter
import torch
import numpy as np
from collections import defaultdict
from utils import ids_to_clean_text
import time
def evaluate(args, model):
    tokenizer = model.tokenizer
    dataset = args.dataset

    if torch.cuda.is_available() and int(args.n_gpu)>1:
        model.model.parallelize()
        device = "cuda"
        model.eval()
    elif int(args.n_gpu)==1:
            model.eval()
            device = "cuda"
            model.to('cuda')
    else:
        print("You need at least one gpu!")
        return
    
    # model.model.parallelize()
    
    #model.to('cuda')

    dataset = Pretrain(dataset=args.dataset, tokenizer=tokenizer, type_path='test', input_length=args.max_input_length, 
                                output_length=args.max_output_length, args=args)


    print('Length of validation data: ',len(dataset))
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,num_workers=4*int(args.n_gpu), pin_memory=True)

    total_cnt = 0
    accuracy_correct_num = 0
    rouge_score = 0 
    f1_score = 0 
    count=0
    total_loss = 0
    batch_cnt = 0
    precisions = defaultdict(list)
    recalls = defaultdict(list)


    prediction_dict = {}
    answer_dict = {}
    if args.output_log != None:
        f = open(args.output_log, 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
    
    predictions = []
    refs = []
    start = time.time()
    for batch in tqdm(iter(loader)):
        #print(batch)
        # output_label = batch["label"].tolist()
        with torch.no_grad():
            batch["source_ids"]=batch["source_ids"].to(device)
            batch["source_mask"]=batch["source_mask"].to(device)
            batch["target_mask"]=batch["target_mask"].to(device)
            batch["target_ids"]=batch["target_ids"].to(device)

            
            outs = model.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=args.max_output_length,
                #num_beams=2,
                early_stopping=True,
            )
            dec = ids_to_clean_text(tokenizer, outs)
            targets = ids_to_clean_text(tokenizer, batch['target_ids']) 
            if args.output_log != None:
                for idx, item in enumerate(dec):
                    wr.writerow([dec[idx], targets[idx]])

    print(f"batch_size: {args.eval_batch_size} / time: {time.time() - start}")
    
    if args.output_log != None:    
        f.close()  