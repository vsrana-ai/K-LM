# coding: utf-8

import os
import brain.config as config
import pkuseg
import numpy as np

from collections import Counter, defaultdict  
from pprint import pprint 
from brain.lm import score 
import shutil 
import typer 
from tqdm import tqdm 
from pathlib import Path 
from collections import namedtuple, OrderedDict 

class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['AI_KG_Small']
    """

    def __init__(self, spo_files, predicate=False, max_entities=config.MAX_ENTITIES): 
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table() 
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)
        self.useable_triples= Counter() 
        self.max_entities = max_entities 
        self.injected_knowledge = defaultdict(list) 

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")    
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = pred + ' ' + obje 
                    else:
                        value = obje

                    if subj in lookup_table.keys():
                        if (value not in lookup_table[subj]):
                            lookup_table[subj].append(value)

                    else:
                        lookup_table[subj] = [value]
             
        print("1.--------Taking a look at Lookup Table--------",type(lookup_table), len(lookup_table))              
        dict_items = lookup_table.items()  
        lookup_table_items = list(dict_items)
        print("Printing Lookup Table ---",lookup_table_items[:3])
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for split_sent in split_sent_batch:

            # create tree  
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []

            #To inject triple having subjects as n-gram 
            for (i, token) in enumerate(split_sent):
                double_token = split_sent[i - 1] + \
                    " " + token if i >= 1 else token  #added 6 sentence spilit 
                triple_token = (split_sent[i - 2] + " " + double_token if i >= 2 else token)

                sing_token_kg = self.lookup_table.get(token, [])
                double_token_kg = self.lookup_table.get(double_token, sing_token_kg)
                triple_token_kg = self.lookup_table.get(triple_token,double_token_kg,)   

                injected_token = "" 

                if triple_token in self.lookup_table:            
                    injected_token = triple_token            
                elif double_token in self.lookup_table:          
                    injected_token = double_token               
                elif token in self.lookup_table:                
                    injected_token = token                        

                if injected_token != "":
                    self.useable_triples.update([injected_token]) 

                entities = triple_token_kg[:max_entities]


                if injected_token != "": 

                    self.injected_knowledge[injected_token] = entities  

                sent_tree.append((token, entities))

                token_pos_idx = [pos_idx + 1]   
                token_abs_idx = [abs_idx + 1]  
                abs_idx = token_abs_idx[-1]
                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    ent_pos_idx = [token_pos_idx[-1]]           
                    entities_pos_idx.append(ent_pos_idx)

                    ent_abs_idx = [abs_idx]                   
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    add_word = [word]
                    know_sent += add_word
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = [sent_tree[i][1][j]]
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
        
        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch


#Ranking the triples uing GPT-2 

app = typer.Typer()

@app.command()
def sort_kg_by_lmscore(spo_path: Path):
    bkp_file = spo_path.with_suffix('.spo.orig') 
    if not bkp_file.exists():
        shutil.copy2(spo_path, spo_path.with_suffix('.spo.orig'))
    lookup_table = OrderedDict()
    KGEntry = namedtuple(
        'KGEntry', ['subj', 'pred', 'obje', 'sent', 'lm_score'])
    print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
    with open(spo_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        for line in tqdm(all_lines): 
            try:
                subj, pred, obje = line.strip().split("\t")
            except Exception:
                print("[KnowledgeGraph] Bad spo:", line)
            value = pred + " " + obje
            sent = subj + ' ' + value
            lm_score = score(sent)
            # lm_score = len(sent)
            kge = KGEntry(subj, pred, obje, sent, lm_score)
            if subj in lookup_table.keys():
                if (
                    value
                    not in lookup_table[subj]
                ):
                    lookup_table[subj].append(kge)
            else:
                lookup_table[subj] = [kge]
    all_subj = list(lookup_table.keys())
    for s in all_subj[:5]:
        print(f'Subj: {s} -> {lookup_table[s][:5]}')

    for s in all_subj:
        kgentry_l = lookup_table[s]
        sorted_kgentry_l = sorted(kgentry_l, key=lambda x: x.lm_score)
        lookup_table[s] = sorted_kgentry_l
    for s in all_subj[:5]:
        print(f'Sorted Subj: {s} -> {lookup_table[s][:5]}')

    with open(spo_path, "w", encoding="utf-8") as f:
        for s in all_subj:
            kges = lookup_table[s]
            for kge in kges:
                f.write('\t'.join([kge.subj, kge.pred, kge.obje])+'\n')
    print("[KnowledgeGraph] written spo to {}".format(spo_path))


if __name__ == '__main__':
    app()
