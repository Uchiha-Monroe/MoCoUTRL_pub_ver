import argparse
from tqdm import tqdm
import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import DebertaModel, DebertaTokenizer

from pytorch_metric_learning.losses import NTXentLoss

from prepro_dataset import BookCorpusDataset, collate_fn_for_ctl




def ctl_parse_args():
    parser = argparse.ArgumentParser(description="You shoule set those parameters")
    
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data_path", default="preprocessed_data/preprocessed_data.txt")
    parser.add_argument("--ptm_path", default="ptm/deberta-base")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--lr", default=1e-6)
    parser.add_argument("--temperature", default=0.07)
    parser.add_argument("--alpha", default=0.5, help="coefficient that controls the weights of two losses")
    parser.add_argument("--momentum", default=0.999, help="momentum coefficient")
    parser.add_argument("--ctl_mdls_save_dir", default="trained_model_save_path/contrastive_encoder")


    args = parser.parse_args()
    return args



if __name__ == "__main__":
    # load args
    ctl_args = ctl_parse_args()
    

    # load training data
    bookcorpus_dataset = BookCorpusDataset(data_path="preprocessed_data/data_50.txt")
    bookcorpus_dataloader = DataLoader(
        dataset=bookcorpus_dataset,
        batch_size=ctl_args.batch_size,
        collate_fn=collate_fn_for_ctl
    )


    # load model
    ptm_model = DebertaModel.from_pretrained(ctl_args.ptm_path)


    # create dynamic word list from ptm_model's Embedding Layer
    ptm_embeddings = copy.deepcopy(ptm_model.embeddings)
    # ptm_embeddings is a instance of torch.nn.Module, with its structure listed as follows:
    # DebertaEmbeddings(
    # (word_embeddings): Embedding(50265, 768, padding_idx=0)
    # (LayerNorm): DebertaLayerNorm()
    # (dropout): StableDropout()
    # )

    # dynamic_word_list is a instance of torch.nn.parameter.Parameter or torch.Tensor
    # take deberta_base model as an example, its size is 
    # >> dynamic_word_list.shape
    # torch.Size([50265, 768])
    dynamic_word_list = list(ptm_embeddings.word_embeddings.parameters())[0].to(device=ctl_args.device)

    ptm_tokenizer = DebertaTokenizer.from_pretrained(ctl_args.ptm_path)
    ptm_model.to(device=ctl_args.device)


    # load optimizer and scheduler
    ctl_optimizer = Adam(
        params=ptm_model.parameters(),
        lr=ctl_args.lr
    )
    ctl_scheduler = CosineAnnealingWarmRestarts(
        optimizer=ctl_optimizer,
        T_0=2000,
        eta_min=1e-8,
        verbose=True
    )


    # load loss_fn
    loss_fn_infonce = NTXentLoss(temperature=ctl_args.temperature)
    loss_fn_mse = nn.MSELoss()


    # training
    accumulate_loss = 0
    step = 0
    iters = len(bookcorpus_dataloader)
    for i_epoch in range(ctl_args.epochs):
        t = tqdm(bookcorpus_dataloader)
        for i, data in enumerate(t):
            original_texts = data["original_texts"]
            synonym_texts = data["synonym_texts"]
            antonym_texts = data["antonym_texts"]
            contrastive_words_index = data["contrastive_words_index"]
            
            # calculate loss1: InfoNCE
            all_texts = tuple(original_texts + synonym_texts + antonym_texts)
            inputs = ptm_tokenizer(all_texts, padding="max_length", truncation=True, max_length=200, return_tensors="pt").to(ctl_args.device)
            
            outputs = ptm_model(**inputs)
            representations = torch.mean(input=outputs.last_hidden_state, dim=1)

            ori_labels = torch.arange(ctl_args.batch_size)
            ant_labels = torch.arange(ctl_args.batch_size, ctl_args.batch_size*2)
            all_labels = torch.cat([ori_labels, ori_labels, ant_labels], dim=0)

            loss_infonce = loss_fn_infonce(representations, all_labels)


            # calculate loss2: MSELoss
            if len(contrastive_words_index) != ctl_args.batch_size:
                loss_mse = loss_infonce
            else:
                
                # calculate word representations from encoder model
                ori_words_repr_list = []
                syn_words_repr_list = []
                ant_words_repr_list = []
                for i in range(ctl_args.batch_size):
                    last_hidden_reprs = outputs.last_hidden_state
                    ori_word_repr = last_hidden_reprs[0, contrastive_words_index[i]]
                    syn_word_repr = last_hidden_reprs[ctl_args.batch_size, contrastive_words_index[i]]
                    ant_word_repr = last_hidden_reprs[ctl_args.batch_size * 2, contrastive_words_index[i]]
                    ori_words_repr_list.append(ori_word_repr)
                    syn_words_repr_list.append(syn_word_repr)
                    ant_words_repr_list.append(ant_word_repr)
                ori_words_embs = torch.stack(ori_words_repr_list, dim=0)
                syn_words_embs = torch.stack(syn_words_repr_list, dim=0)
                ant_words_embs = torch.stack(ant_words_repr_list, dim=0)
                all_batch_embs = torch.cat((ori_words_embs, syn_words_embs, -ant_words_embs), dim=0)

                # get the word representations from dynamic dictionary
                input_ids = inputs["input_ids"]
                ori_gold_list = []
                syn_gold_list = []
                ant_gold_list = []
                for i in range(ctl_args.batch_size):
                    ori_gold_idx = input_ids[0, contrastive_words_index[i]].item()
                    syn_gold_idx = input_ids[ctl_args.batch_size, contrastive_words_index[i]].item()
                    ant_gold_idx = input_ids[ctl_args.batch_size * 2, contrastive_words_index[i]].item()
                    ori_gold_list.append(ori_gold_idx)
                    syn_gold_list.append(syn_gold_idx)
                    ant_gold_list.append(ant_gold_idx)
                all_gold_idxs = torch.tensor(ori_gold_list + syn_gold_list + ant_gold_list, dtype=torch.long).to(device=ctl_args.device)   
                all_batch_golds = torch.index_select(dynamic_word_list, dim=0, index=all_gold_idxs)
                
                # calculate loss2
                loss_mse = loss_fn_mse(all_batch_embs, all_batch_golds)
            
            # aggregate losses
            loss = ctl_args.alpha * loss_infonce + (1-ctl_args.alpha) * loss_mse
            
            # update encoder's params by BP 
            # if step < 100:
            #     loss.backward(retain_graph=True)
            # else:
            #     loss.backward()
            loss.backward(retain_graph=True)
            ctl_optimizer.step()
            ctl_scheduler.step(i_epoch + i / iters)

            # update dynamic dictionary's params by momentum update
            current_emb = list(ptm_model.embeddings.word_embeddings.parameters())[0]
            dynamic_word_list = ctl_args.momentum*dynamic_word_list + (1-ctl_args.momentum)*current_emb

            temp_loss = loss.item()
            accumulate_loss += temp_loss
            step += 1
            mean_loss = accumulate_loss / step
            t.set_description(f'epoch: {i_epoch+1}, mean_loss: {mean_loss:.06}, temp_loss: {temp_loss:.06}')

            if step % 1000 == 0 and mean_loss < 0.8:
            # if step % 50 == 0:
                path_ctl_model = os.path.join(ctl_args.ctl_mdls_save_dir, f"meanloss_{mean_loss:.04}.pt")
                print(f"saving models to {ctl_args.ctl_mdls_save_dir}...")
                torch.save(ptm_model, path_ctl_model)
                print(f"Done!")


            # print("Done")

