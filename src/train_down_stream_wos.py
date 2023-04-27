import argparse
from tabnanny import verbose
from tqdm import tqdm
import copy
import os

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

from transformers import DebertaModel, DebertaTokenizer

from sklearn.metrics import precision_recall_fscore_support

from prepro_dataset import WOSDataset
from models import LinearMacroClassfier



def wos_train_parse_args():
    parser = argparse.ArgumentParser(description="Set parameters for WOS training")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_path", default="downstream_datasets/WOS_dataset/WebOfScience/Meta-data/Data.xlsx")
    parser.add_argument("--ptm_path", default="ptm/deberta-base")
    parser.add_argument("--trained_model_dir", default="trained_model_save_path/supervised_wos")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", default=3)
    parser.add_argument("--lr", default=3e-7)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    wostrain_args = wos_train_parse_args()

    # load training data
    print(f"loading data...")
    wos_train_dataset = WOSDataset(split="train")
    wos_train_dataloader = DataLoader(
        dataset=wos_train_dataset,
        batch_size=wostrain_args.batch_size,
        shuffle=True
    )

    wos_valid_dataset = WOSDataset(split="valid")
    wos_valid_dataloader = DataLoader(
        dataset=wos_valid_dataset,
        batch_size=wostrain_args.batch_size
    )

    wos_test_dataset = WOSDataset(split="test")
    wos_test_dataloader = DataLoader(
        dataset=wos_test_dataset,
        batch_size=wostrain_args.batch_size
    )
    print(f"done!")


    # load model
    print(f"loading model...")
    encoder = DebertaModel.from_pretrained(wostrain_args.ptm_path, return_dict=True).to(device=wostrain_args.device)
    ptm_tokenizer = DebertaTokenizer.from_pretrained(wostrain_args.ptm_path)
    clf = LinearMacroClassfier().to(device=wostrain_args.device)
    encoder.train()
    clf.train()
    print(f"done!")
    

    # load optimizer and scheduler
    wos_optimizer = Adam(
        [
            {"params": encoder.parameters(), "lr": wostrain_args.lr},
            {"params": clf.parameters(), "lr":wostrain_args.lr}
        ]
    )
    wos_scheduler = CosineAnnealingWarmRestarts(wos_optimizer, T_0=1500, eta_min=1e-8, verbose=True)


    # 
    accumulate_loss = 0
    step = 0
    iters = len(wos_train_dataloader)
    for i_epoch in range(wostrain_args.epochs):
        t = tqdm(wos_train_dataloader)
        for i, data in enumerate(t):
            texts, *_, y, domain = data
            y = y.to(device=wostrain_args.device)
            
            inputs = ptm_tokenizer(texts, padding="max_length", truncation=True, max_length=200, return_tensors="pt").to(device=wostrain_args.device)
            outputs = encoder(**inputs)
            representations = torch.mean(input=outputs.last_hidden_state, dim=1)
            predict = clf(representations)

            # calculate loss
            loss_fn = F.cross_entropy
            loss = loss_fn(predict, y)
            loss.backward()
            wos_optimizer.step()
            wos_scheduler.step(i_epoch + i / iters)

            temp_loss = loss.item()
            accumulate_loss += temp_loss
            step += 1
            mean_loss = accumulate_loss / step
            t.set_description(f'epoch: {i_epoch+1}, mean_loss: {mean_loss:.06}, temp_loss: {temp_loss:.06}')

            
            # evaluate every 100 steps
            if step % 1000 == 0:
                print(f"start evaluating on valid set...")
                encoder.eval()
                clf.eval()
                all_labels = []
                all_predicts = []
                for data in wos_valid_dataloader:
                    texts, *_, y, domain = data
                    inputs = ptm_tokenizer(texts, padding="max_length", truncation=True, max_length=200, return_tensors="pt").to(device=wostrain_args.device)
                    outputs = encoder(**inputs)
                    representations = torch.mean(input=outputs.last_hidden_state, dim=1)
                    logits = clf(representations)
                    predict = torch.argmax(logits, dim=1, keepdim=False).cpu().tolist()
                    label = y.cpu().tolist()

                    all_labels.extend(label)
                    all_predicts.extend(predict)

                predicts_np = np.array(all_predicts)
                labels_np = np.array(all_labels)
               
                macro_p, macro_r, macro_f1_score, *_ = precision_recall_fscore_support(
                    y_true=labels_np,
                    y_pred=predicts_np,
                    average="macro"
                )
                micro_p, micro_r, micro_f1_score, *_ = precision_recall_fscore_support(
                    y_true=labels_np,
                    y_pred=predicts_np,
                    average="micro"
                )

                print(f"evaluate on validset: macro_p:{macro_p}, macro_r:{macro_r}, macro_f1_score:{macro_f1_score}")
                print(f"evaluate on validset: micro_p:{micro_p}, micro_r:{micro_r}, micro_f1_score:{micro_f1_score}")

                # after evaluation, turn the model to train mode
                encoder.train()
                clf.train()

                # save the model if possible
                if micro_f1_score > 0.88:
                    print(f"macro_f1_score on valid set is {macro_f1_score}, save the models")
                    path_encoder_model = os.path.join(wostrain_args.trained_model_dir, f"encoder_macrof1_{macro_f1_score:.04}.pt")
                    path_clf_model = os.path.join(wostrain_args.trained_model_dir, f"clf_macrof1_{macro_f1_score:.04}.pt")
                    print(f"saving models to {wostrain_args.trained_model_dir}...")
                    torch.save(encoder, path_encoder_model)
                    torch.save(clf, path_clf_model)
                    print(f"Done!")






            

            

