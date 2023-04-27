import os
from random import choice, randint
from tqdm import tqdm, trange

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizer, DebertaTokenizer, DebertaModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

text_data_path = "downstream_datasets/WOS_dataset/WebOfScience/Meta-data/test.csv"
data_df = pd.read_csv(text_data_path)

# load models
encoder_path = "trained_model_save_path/contrastive_encoder/meanloss_1.516.pt"
tokenizer_path = "ptm/deberta-base"
with open(encoder_path, "rb") as fm:
    encoder = torch.load(fm).to(DEVICE)
tokenizer = DebertaTokenizer.from_pretrained(tokenizer_path)
# ptm = BertModel.from_pretrained(tokenizer_path).to(DEVICE)

if_evaluate_each_token = True
count = 0
all_feas_list = []
for i in trange(data_df.shape[0]):
    
    # if data_df.loc[i, "Y_major"] != 6:
    #     continue
    
    text = data_df.loc[i, "Abstract"]
    inputs = tokenizer(text, truncation=True, max_length=200, return_tensors="pt").to(DEVICE)
    
    # use choosed encoder to get the input's representation
    outputs = encoder(**inputs)
    
    if if_evaluate_each_token == False:
        mean_pool_fea = torch.mean(input=outputs.last_hidden_state, dim=1, keepdim=False)  
        current_fea = mean_pool_fea.detach().cpu()
        if count == 0:
            all_feas = current_fea
        else:
            all_feas = torch.cat(tensors=(all_feas, current_fea), dim=0)
        count += 1
    else:
        maxlen = outputs.last_hidden_state.shape[1]
        for j in range(3):
            rd_idx = randint(0, maxlen-1)
            random_chosen_fea = outputs.last_hidden_state[:, rd_idx].detach().cpu()
            all_feas_list.append(random_chosen_fea)

all_feas = torch.cat(tensors=all_feas_list, dim=0)

all_feas_np = all_feas.numpy()
np.save("jptnotes/draw_wos_data/deberta_ctl_test_alllabel_randomchoice.npy", all_feas_np)

print("Done!")