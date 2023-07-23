import transformers
import torch
import pandas as pd
from tqdm import tqdm
model_path = 'google/mt5-base'
df = pd.read_excel('test.xlsx')

model = transformers.MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

#for name, param in model.named_parameters():
#    print(name)

checkpoint_path = 'ckpt/mt5_base_papago_v4/epoch_4.pt'
print(checkpoint_path)
loaded_ckpt = torch.load(checkpoint_path)
loaded_model={}
for key, value in loaded_ckpt.items():
    #print(key)
    loaded_model[key] = value
model.load_state_dict(loaded_model, strict=False)
out_list = []
for idx, row in df.iterrows():
    input_ids = tokenizer(
        row['오류조합주소'],
        return_tensors="pt"
    ).input_ids
    
    output = model.generate(input_ids=input_ids)
    
    answer = tokenizer.decode(output[0]).replace('<pad> ', '').strip()
    answer = answer.replace('</s>', '').strip()
    out_list.append([row['오류조합주소'], row['정답(도로명주소)'], answer])
    print(answer)

hey = pd.DataFrame(out_list)
column_label = ['오류조합주소', '정답(도로명주소)', '모델아웃풋']

hey.to_csv("result/base_papago_v4_epoch4.csv", index=False, header=column_label)