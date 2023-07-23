
import pytorch_lightning as pl
from transformers import AutoTokenizer, MT5ForConditionalGeneration, Adafactor
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from datasets import Pretrain
import torch
from torch.optim import AdamW
import os
import functools
from utils import ids_to_clean_text

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class MT5_MODEL(pl.LightningModule):
    def __init__(self, args):
        super(MT5_MODEL, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        self.epoch = 0
        self.model = MT5ForConditionalGeneration.from_pretrained(args.model_name_or_path)            

    def get_dataset(self, dataset, tokenizer, type_path, args):
        dataset = Pretrain(dataset=dataset, tokenizer=tokenizer, type_path=type_path, input_length=args.max_input_length, 
                                output_length=args.max_output_length, args=args)
        return dataset
        
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
    
    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids= batch["source_ids"],
            attention_mask= batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask= batch["target_mask"]
        )
        loss = outputs[0]
    
        return loss

    def _generative_step(self, batch, batch_idx, dataloader_idx):  
        keys = batch["data_label"][0]
        output_label = batch["label"].tolist()

        accuracy_correct_num=0
        rouge_score=0
        total_cnt = 0
        if 'dev' in keys or (keys == 'target' and self.args.eval_with_prob == False):
            outs = self.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=self.args.max_output_length,
                num_beams=2,
                early_stopping=True,
            )
            dec = ids_to_clean_text(self.tokenizer, outs)
        else:
            prob_list = []
            with torch.no_grad():
                for index in range(len(batch["option_list"])):
                    option = batch["option_list"]
                    option_ = self.tokenizer.batch_encode_plus(option[index], max_length=self.args.max_output_length,
                                                    padding=True, truncation=True, return_tensors="pt")
                    lm_labels = option_["input_ids"].expand(len(batch['source_ids']), -1)
                    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                    outputs = self.model(
                        input_ids=batch["source_ids"].cuda(),
                        attention_mask=batch["source_mask"].cuda(),
                        labels=lm_labels.cuda(),
                        decoder_attention_mask=option_["attention_mask"].cuda()
                    )
                    #print("outputs", outputs[0])
                    logits = option_["attention_mask"].cuda().unsqueeze(-1) * torch.log_softmax(outputs.logits, dim=-1)
                    lm_labels=lm_labels.cuda().unsqueeze(-1)
                    seq_token_log_prob=torch.zeros(lm_labels.shape)
                    #print(seq_token_log_prob.shape, logits.shape, lm_labels.shape)
                    for i in range(lm_labels.shape[0]):
                        for j in range(lm_labels.shape[1]):
                            seq_token_log_prob[i][j][0] = logits[i][j][lm_labels[i][j][0]]
                    seq_log_prob = seq_token_log_prob.squeeze(dim=-1).sum(dim=-1)
                    prob_list.append(seq_log_prob)
                concat = torch.cat(prob_list).view(-1,len(batch['source_ids']))
                #print(concat)
                predictions = concat.argmax(dim=0)
                dec = [batch["option_list"][i.item()][elem_num] for elem_num, i in enumerate(predictions)]

        preds_list = []
        targets_list = []
        total_cnt+=len(batch['source_ids'])
        targets = ids_to_clean_text(self.tokenizer, batch['target_ids']) 

        acc_score = sum(list(map(lambda v: v[0] ==v[1],zip(predictions,output_label)))) 
                
        return acc_score, total_cnt


    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.args.optimizer_type == 'adafactor':
            optimizer = Adafactor(optimizer_grouped_parameters, lr=self.args.learning_rate, scale_parameter=False, relative_step=False)
        else: 
            optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)

        if self.args.use_lr_scheduling:
            len_data = len(self.train_dataloader())
            denomniator = (self.args.n_gpu * self.args.gradient_accumulation_steps)
            steps_per_epoch = ( len_data // denomniator ) + 1
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.args.num_train_epochs, anneal_strategy='linear', cycle_momentum=False)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        accs_dict = {}
        accuracy, cnt = self._generative_step(batch, batch_idx, dataloader_idx)
        return {batch['data_label'][0]: [accuracy, cnt]}
    
    def test_step(self, batch, batch_idx):
        for keys in batch.keys():
            self._generative_step(batch, keys, batch_idx)

    def on_train_epoch_end(self):
        param_dict = {}
        os.makedirs(f'{self.args.output_dir}', exist_ok=True)
        for name, param in self.model.named_parameters():
            param_dict[name]=param.clone().detach().cpu()
        torch.save(param_dict, f'{self.args.output_dir}/epoch_{self.epoch}.pt') 
        self.epoch += 1
        
    def on_validation_epoch_end(self, validation_step_outputs): 
        score_dict = {}
        score = 0
        
        validation_step_outputs_gather = self.all_gather(validation_step_outputs)

        for output in validation_step_outputs_gather:
            for key, [accs, cnt] in output.items():
                if key not in score_dict.keys():
                    score_dict[key]=[accs, cnt]
                else:
                    old_acc, old_cnt = score_dict[key]
                    score_dict[key] = [old_acc + accs, old_cnt + cnt]    

        self.val_count += 1     

    def train_dataloader(self):
        train_dataset = self.get_dataset(dataset=self.args.dataset, tokenizer=self.tokenizer, type_path="train", args=self.args)
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.args.train_batch_size, drop_last=False, num_workers=self.args.num_workers)
        return dataloader

    def val_dataloader(self):
        return []
        val_dataset_1 = self.get_dataset(dataset=self.args.dataset, tokenizer=self.tokenizer, type_path="dev", args=self.args)
        return DataLoader(val_dataset_1, batch_size=self.args.eval_batch_size, drop_last=True, num_workers=self.args.num_workers)
            
    def test_dataloader(self):
        return self.val_dataloader()