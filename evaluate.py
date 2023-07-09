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
    for batch in tqdm(iter(loader)):
        #print(batch)
        # output_label = batch["label"].tolist()
        with torch.no_grad():
            batch["source_ids"]=batch["source_ids"].to(device)
            batch["source_mask"]=batch["source_mask"].to(device)
            batch["target_mask"]=batch["target_mask"].to(device)
            batch["target_ids"]=batch["target_ids"].to(device)

            # t0 evaluation method - select option with higer probability as answer
            if args.eval_with_prob:
                output_label = batch["label"].tolist()
                prob_list = []
                prob_list_calibrate = []
                for index in range(len(batch["option_list"])):
                    #print(batch["option_list"])
                    option = batch["option_list"]
                    option_ = tokenizer.batch_encode_plus(option[index], max_length=args.max_output_length,
                                                    padding=True, truncation=True, return_tensors="pt")
                    option_["input_ids"]=option_["input_ids"].to(device)
                    option_["attention_mask"]=option_["attention_mask"].to(device)
                    lm_labels = option_["input_ids"].expand(len(batch['source_ids']), -1)
                    lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                    outputs = model.model(
                        input_ids=batch["source_ids"],
                        attention_mask=batch["source_mask"],
                        labels=lm_labels,
                        decoder_attention_mask=option_["attention_mask"]
                    )

                    logits = option_["attention_mask"].unsqueeze(-1) * torch.log_softmax(outputs.logits, dim=-1)

                    if args.calibrate:
                        prompt_temp = tokenizer.batch_encode_plus([""]*len(batch["option_list"][0]), max_length=128, 
                                                        padding='max_length', truncation=True, return_tensors="pt")

                        outputs_calibrate = model.model(
                            input_ids=prompt_temp["input_ids"].cuda(),
                            attention_mask=prompt_temp["attention_mask"].cuda(),
                            labels=lm_labels.cuda(),
                            decoder_attention_mask=option_["attention_mask"].cuda()
                        )
                        logits_calibrate = option_["attention_mask"].cuda().unsqueeze(-1) * torch.log_softmax(outputs_calibrate.logits, dim=-1)
                    

                    lm_labels=lm_labels.unsqueeze(-1)
                    seq_token_log_prob=torch.zeros(lm_labels.shape)
                    #print(seq_token_log_prob.shape, logits.shape, lm_labels.shape)
                    for i in range(lm_labels.shape[0]):
                        for j in range(lm_labels.shape[1]):
                            seq_token_log_prob[i][j][0] = logits[i][j][lm_labels[i][j][0]]
                    seq_log_prob = seq_token_log_prob.squeeze(dim=-1).sum(dim=-1)
                    prob_list.append(seq_log_prob)
                    if args.calibrate:
                        seq_token_log_prob_calibrate=torch.zeros(lm_labels.shape)
                        for i in range(lm_labels.shape[0]):
                            for j in range(lm_labels.shape[1]):
                                seq_token_log_prob_calibrate[i][j][0] = logits_calibrate[i][j][lm_labels[i][j][0]]
                        seq_log_prob_calibrate = seq_token_log_prob_calibrate.squeeze(dim=-1).sum(dim=-1)
                        prob_list_calibrate.append(seq_log_prob_calibrate)
                concat = torch.cat(prob_list).view(-1,len(batch['source_ids']))
                if args.calibrate:
                    calibrate_concat = torch.cat(prob_list_calibrate).view(-1,len(batch['source_ids']))
                    #print(concat.shape, calibrate_concat.shape)
                    concat = concat - calibrate_concat
                predictions = concat.argmax(dim=0)
                dec = [batch["option_list"][i.item()][elem_num] for elem_num, i in enumerate(predictions)]
            else:
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
                predictions.append(dec[0])
                refs.append(targets[0])
            total_cnt+=len(batch['source_ids'])
            if args.eval_with_prob:
                
                #print(predictions, output_label)
                accuracy_correct_num += sum(list(map(lambda v: v[0] ==v[1],zip(predictions,output_label)))) 

    print(f'Number of total validation data: {total_cnt}')

    print('Number of correct predictions: {: .0f}. Percentage : {: .4f}'.format(accuracy_correct_num, accuracy_correct_num/total_cnt))
    final_score = float(accuracy_correct_num / total_cnt)

    if args.checkpoint_path == '':
        first_config = args.model_name_or_path
    else:
        first_config = args.checkpoint_path
    if args.output_log != None:
        wr.writerow([first_config, args.dataset, final_score])
    if args.output_log != None:    
        f.close()  