import os
from random import choice

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer

class BookCorpusDataset(Dataset):
    """
    Dataset of BookCorpus, this class has two types of usage:
      1. If the data are raw and in the format of each line in a sentence, then this can preprocess the
         raw data to the format that are suitable to the model
      2. If the data is preprocessed, this class can directly load it for majorsequent use
    
    The preprocessed data contains original, synonym and antonym texts.
    """
    def __init__(
        self, 
        data_path: str = "preprocessed_data/preprocessed_data.txt", 
    ) -> None:
        """
        Args:
            data_path: path of data, can either be raw data or preprocessed data.
            model_path: path of pretrained model, used to load the tokenizer of the specified model
            do_preprocess: bool, if set to True the __init__ func will preprocess the raw data and save it to disk.
        """
        super().__init__()
        

        # directly load preprocessed data
        print("Directly reading preprocessed data...")
        self.all_original_texts = []
        self.all_synonym_texts = []
        self.all_antonym_texts = []
        with open(data_path, "r", encoding="utf-8") as fr:
            all_lines = fr.readlines()
            skip_firstrow = 0
            for single_line in tqdm(all_lines):
                if skip_firstrow == 0:
                    skip_firstrow += 1
                    continue
                ori, syn, ant = single_line.strip().split("\t")
                self.all_original_texts.append(ori)
                self.all_synonym_texts.append(syn)
                self.all_antonym_texts.append(ant)


    # if data is raw, preprocess it
    @staticmethod
    def preprocess(
        ptm_tokenizer_path="ptm/deberta-base",
        raw_data_path="books1/tiny_text_50.txt",
        output_path="preprocessed_data/data_50.txt"
    ):
        print("The data is raw and now preprocess it...")
        tokenizer = DebertaTokenizer.from_pretrained(ptm_tokenizer_path)
        with open(raw_data_path, "r", encoding="utf-8")as fr:
            all_lines = fr.readlines()
            num_lines = len(all_lines)
            
            fw = open(output_path, "w", encoding="utf-8")
            fw.write("original_text\tsynonym_text\tantonym_text\n")

            # traverse all the lines of the data and preprocess
            for line in tqdm(all_lines):
                skip_flag = 0
                line = line.strip()
                
                line_words_list = word_tokenize(line)
                if len(line_words_list) < 25 or len(line_words_list) > 500:
                    continue

                pos_tagged = pos_tag(line_words_list)
                all_adj_words = []
                
                for word, pos in pos_tagged:
                    if pos == "JJ":
                        all_adj_words.append(word)
                # If there is no adjective in this sentence, go through it
                if len(all_adj_words) == 0:
                    continue

                # synonym and antonym majorstitution
                # random choose a word in the single sentence and use a synonym word to majorstitute it
                while(True):
                    ori_word = choice(all_adj_words) # random choose a single word from the original sentence
                    ori_index = line_words_list.index(ori_word)
                    
                    synonyms = [] # save all the synonyms of the original word
                    antonyms = [] # save all the antonyms of the original word

                    for syn in wordnet.synsets(ori_word):
                        for lm in syn.lemmas():
                            synonyms.append(lm.name())
                    
                    for syn in wordnet.synsets(ori_word):
                        for lm in syn.lemmas():
                            if lm.antonyms():
                                antonyms.append(lm.antonyms()[0].name())

                    
                    if len(synonyms) == 0 or len(antonyms) == 0:
                        skip_flag = 1
                        break
                    else:
                        syn_word = choice(synonyms)
                        ant_word = choice(antonyms)
                        break
                if skip_flag == 1:
                    continue
                
                ori_line = " ".join(line_words_list)

                syn_words_list = line_words_list.copy()
                syn_words_list[ori_index] = syn_word
                syn_line = " ".join(syn_words_list)

                ant_words_list = line_words_list.copy()
                ant_words_list[ori_index] = ant_word
                ant_line = " ".join(ant_words_list)

                # write preprocessed text to disk
                fw.write(f"{ori_line}\t{syn_line}\t{ant_line}\n")
            fw.close()
        

    def __getitem__(self, index):
        return self.all_original_texts[index], self.all_synonym_texts[index], self.all_antonym_texts[index]


    def __len__(self):
        return len(self.all_original_texts)
                
def collate_fn_for_ctl(
    data,
    tokenizer_model_path="./ptm/deberta-base/"
):
    """
    This func can organize the dataset and return as a dict format:
    {
        "original_texts": List[str],
        "synonym_texts": List[str],
        "antonym_texts": List[str],
        "contrastive_words_index": List[int], # The index of changed token after the pretrained tokenization
    }
    
    """
    model_tokenizer = DebertaTokenizer.from_pretrained(tokenizer_model_path)
    
    original_texts, synonym_texts, antonym_texts, contrastive_words_index = [], [], [], []
    
    for unit in data:
        ori, syn, ant = unit
        original_texts.append(ori)
        synonym_texts.append(syn)
        antonym_texts.append(ant)
        
        ori_input_ids = model_tokenizer(ori, padding="max_length", truncation=True, max_length=200, return_tensors="pt")["input_ids"].squeeze(0)
        ant_input_ids = model_tokenizer(ant, padding="max_length", truncation=True, max_length=200, return_tensors="pt")["input_ids"].squeeze(0)
        min_length = min(len(ori_input_ids), len(ant_input_ids))
        for i in range(min_length):
            if ori_input_ids[i] == ant_input_ids[i]:
                continue
            else:
                contrastive_words_index.append(i)
        if len(contrastive_words_index) != len(original_texts):
            continue
    
    return {
        "original_texts": original_texts,
        "synonym_texts": synonym_texts,
        "antonym_texts": antonym_texts,
        "contrastive_words_index": contrastive_words_index
    }

class WOSDataset(Dataset):
    """
    Dataset of downstream dataset Web of Science.
    """
    def __init__(self, split:str, datadir="downstream_datasets/WOS_dataset/WebOfScience/Meta-data/"):
        super().__init__()
        
        datapath = os.path.join(datadir, (split + ".csv"))
        raw_df = pd.read_csv(datapath)
        
        self.field_y = raw_df.loc[:, "Y"]
        self.field_area = raw_df.loc[:, "area"]
        self.field_y1 = raw_df.loc[:, "Y_major"]
        self.field_domain = raw_df.loc[:, "area_major"]
        self.field_abstract = raw_df.loc[:, "Abstract"]

    
    @staticmethod
    def preprocess_split_data(
        rawdata_path = "downstream_datasets/WOS_dataset/WebOfScience/Meta-data/Data.xlsx",
        output_dir = "downstream_datasets/WOS_dataset/WebOfScience/Meta-data/",
        train_ratio = 0.8,
        valid_ratio = 0.1,
        test_ratio = 0.1
    ):
        print("start reading and spliting raw data...")
        raw_df = pd.read_excel(rawdata_path)

        raw_df = raw_df.rename(columns={"Y1": "Y_major", "Domain": "area_major"})

        num_sample = raw_df.shape[0]
        train_valid_split = int(train_ratio*num_sample)
        valid_test_split = int((train_ratio+valid_ratio)*num_sample)

        trainset = raw_df.loc[:train_valid_split, ["Y", "area", "Y_major", "area_major", "Abstract"]]
        validset = raw_df.loc[train_valid_split:valid_test_split, ["Y", "area", "Y_major", "area_major", "Abstract"]]
        testset = raw_df.loc[valid_test_split: , ["Y", "area", "Y_major", "area_major", "Abstract"]]

        trainset.to_csv(os.path.join(output_dir, "train.csv"))
        validset.to_csv(os.path.join(output_dir, "valid.csv"))
        testset.to_csv(os.path.join(output_dir, "test.csv"))
        

    def __len__(self):
        return len(self.field_y)


    def __getitem__(self, index):
        return self.field_abstract[index], self.field_y[index], self.field_area[index], self.field_y1[index], self.field_domain[index]



if __name__ == "__main__":
    # aa = BookCorpusDataset(data_path="preprocessed_data/preprocessed_data.txt", do_preprocess=False)
    # BookCorpusDataloader = DataLoader(dataset=aa, batch_size=4, collate_fn=collate_fn_for_ctl)
    # BookCorpusDataset.preprocess(raw_data_path="books1/tiny_text_50.txt")
    # bb = BookCorpusDataset()
    WOSDataset.preprocess_split_data()
    print("Done!")