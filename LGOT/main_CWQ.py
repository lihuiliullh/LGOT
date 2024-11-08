import argparse
import json
import logging
import os
import random
import collections
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from kg_model import KGReasoning
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
from chatGPT import get_answer_chatGPT
from relation2description import *
import traceback


prompt_example = """
I will give you a question. Please output at most 10 answers for this question in a string format separated by commas. Do not give multiple answers.

Examples:
Q: movie interstellar starred by who?
A: "Matthew McConaughey", "Anne Hathaway", "Jessica Chastain", "Bill Irwin", "Ellen Burstyn", "Michael Caine"

Q: Which movie starred Tom Hanks?
A: "Forrest Gump", "A Man Called Otto", "Cast Away", "The Green Mile", "The Terminal", "Saving Private Ryan"

Q: who stars movie Titanic or Interstellar?
A: "Leonardo DiCaprio", "Kate Winslet", "Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"

Q: """


def answers_by_chatGPT(head_set, rel, relation2description):
    prompt = relation2description[rel]
        
    candidate_entities = list(head_set)
    if len(candidate_entities) <= 2:
        ents = " or ".join(candidate_entities)
    else:
        ents = ", ".join(candidate_entities[: -1]) + ", or " + candidate_entities[-1]
    
    prompt = prompt.replace("entity1", ents)
    
    prompt = prompt_example + prompt + '\nA:'
    
    response = get_answer_chatGPT(prompt)
    return response


def answer_by_kg(embedding, rel_idx, kg_model):
    
    r_embedding = kg_model.get_relation_embedding(rel_idx)
    embedding, r_argmax = kg_model.relation_projection(embedding, r_embedding, False)
    
    return embedding
        
    
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--do_cp', action='store_true', help="do cardinality prediction")
    parser.add_argument('--path', action='store_true', help="do interpretation study")

    parser.add_argument('--train', action='store_true', help="do test")
    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--kbc_path', type=str, default=None, help="kbc model path")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int, help="used to speed up torch.dataloader")
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--fraction', type=int, default=1, help='fraction the entity to save gpu memory usage')
    parser.add_argument('--thrshd', type=float, default=0.001, help='thrshd for neural adjacency matrix')
    parser.add_argument('--neg_scale', type=int, default=1, help='scaling neural adjacency matrix for negation')
    
    #parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=12345, type=int, help="random seed")
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    return parser.parse_args(args)



def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True


def read_triples(filenames, nrelation, datapath, ent2id, rel2id):
    adj_list = [[] for i in range(nrelation)]
    edges_all = set() # all edges of train + valid + test
    edges_vt = set() # all edges of valid + test
    for filename in filenames:
        with open(filename) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                h = ent2id[h]
                t = ent2id[t]
                r = rel2id[r]
                adj_list[int(r)].append((int(h), int(t)))
    for filename in ['valid', 'test']:
        with open(os.path.join(datapath, filename)) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                h = ent2id[h]
                t = ent2id[t]
                r = rel2id[r]
                edges_all.add((int(h), int(r), int(t)))
                edges_vt.add((int(h), int(r), int(t)))
    with open(os.path.join(datapath, "train")) as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            h = ent2id[h]
            t = ent2id[t]
            r = rel2id[r]
            edges_all.add((int(h), int(r), int(t)))

    return adj_list, edges_all, edges_vt



def get_relation2description():
    relation2description = pickle.load(open("chatGPT_CWQ_relation2text.pkl", "rb"))
    for k, v in relation2description.items():
        v = json.loads(v)['answer']
        
        if "Australia's state of Victoria" in v:
            v = v.replace("Australia's state of Victoria", "entity1")
        
        if "entity2" in v:
            v = v.replace("entity2", "entity1")
        
        if "the Philippines Campaign (1941–1942)" in v:
            v = v.replace("the Philippines Campaign (1941–1942)", "entity1")
        
        if "Who won the sports championship event" in v:
            v = "Who won the sports championship event entity1" 
        
        if "D-Day World War II" in v:
            v = v.replace("D-Day World War II", "entity1")
        
        if "Despicable Me: Part 2" in v:
            v = v.replace("Despicable Me: Part 2", "entity1")
        
        if "Lazar Markovitch Zamenhof" in v:
            v = v.replace("Lazar Markovitch Zamenhof", "entity1")
        
        if "Striptease (Danity Kane song)" in v:
            v = v.replace("Striptease (Danity Kane song)", "entity1")
        
        if "President Buchanan" in v:
            v = v.replace("President Buchanan", "entity1")
        
        if "regular TV appearances" in v:
            v = "Who is the actor with regular TV appearances in entity1?"
        
        if "m.0j_nxfw" in v:
            v = v.replace("m.0j_nxfw", "entity1")
        
        if "Education in Iowa" in v:
            v = v.replace("Education in Iowa", "entity1")
        
        if "Missouri River (United States)" in v:
            v = v.replace("Missouri River (United States)", "entity1")
        
        if "Who was nominated for an award" in v:
            v = "Who was nominated for an award entity1?"
        
        if "composed the music composition" in v:
            v = "Who composed the music composition entity1?"
        
        if "The Russian federation" in v:
            v = v.replace("The Russian federation", "entity1")
        
        if "Mountain Daylight Time" in v:
            v = v.replace("Mountain Daylight Time", "entity1")
        
        if "Yankee land" in v:
            v = v.replace("Yankee land", "entity1")
        
        if "Brewers Roster" in v:
            v = v.replace("Brewers Roster", "entity1")
        
        if "Al memzar, dubai" in v:
            v = v.replace("Al memzar, dubai", "entity1")
        
        if "U.S. Representatives" in v:
            v = v.replace("U.S. Representatives", "entity1")
        
        if "Jewish ancestry" in v:
            v = v.replace("Jewish ancestry", "entity1")
        
        if "Aubrey Morgan O'Day" in v:
            v = v.replace("Aubrey Morgan O'Day", "entity1")
        
        if "Games of the XXVI Olympiad" in v:
            v = v.replace("Games of the XXVI Olympiad", "entity1")
        
        if "Brewers Roster" in v:
            v = v.replace("Brewers Roster", "entity1")
        
        if "Dayton Arrows" in v:
            v = v.replace("Dayton Arrows", "entity1")
        
        if "Justin Bieber" in v:
            v = v.replace("Justin Bieber", "entity1")
        
        if "Joàn Miró" in v:
            v = v.replace("Joàn Miró", "entity1")
        
        if "Mall of the emirates" in v:
            v = v.replace("Mall of the emirates", "entity1")
        
        if "Sweyn II" in v:
            v = v.replace("Sweyn II", "entity1")
        
        if "Mike Tomlin" in v:
            v = v.replace("Mike Tomlin", "entity1")
        
        if "Fuxi" in v:
            v = v.replace("Fuxi", "entity1")
        
        if "Tao Teh Ching" in v:
            v = v.replace("Tao Teh Ching", "entity1")
        
        if "Pittsburg Steelers" in v:
            v = v.replace("Pittsburg Steelers", "entity1")
        
        if "AL West" in v:
            v = v.replace("AL West", "entity1")
        
        if "UN/LOCODE:MXMEX" in v:
            v = v.replace("UN/LOCODE:MXMEX", "entity1")
        
        if "Nicollet Island" in v:
            v = v.replace("Nicollet Island", "entity1")
        
        if "Taoshi" in v:
            v = v.replace("Taoshi", "entity1")
        
        if "Franklin Stove" in v:
            v = v.replace("Franklin Stove", "entity1")
        
        if "('m.04c5z02', 'm.04c5z02')" in v:
            v = v.replace("('m.04c5z02', 'm.04c5z02')", "entity1")
        
        if "entity1" not in v:
            print(v)
    
        relation2description[k] = v
    return relation2description


def main(args):
    set_global_seed(args.seed)
    device = args.device
    
    freebase_id2name = pickle.load(open(args.freebase, "rb"))
    question = pickle.load(open(args.qa_file, "rb"))

    result_file = args.result_file
    relation2description = get_relation2description()
    
    args.fraction = 500
    args.thrshd = 0.01
    
    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    
    global id2ent, id2rel
    ent2id = {}
    rel2id = {}
    id2ent = {}
    id2rel = {}
    lowerEnt2originalEnt = {}
    with open('%s/ent_id'%args.data_path_id_information, 'r') as f:
        for ele in f.readlines():
            eles = ele.strip().split("\t")
            if len(eles) < 2:
                continue
            ent2id[eles[0]] = int(eles[1])
            id2ent[int(eles[1])] = eles[0]
            lowerEnt2originalEnt[eles[0].lower()] = eles[0]

        
    with open('%s/rel_id'%args.data_path_id_information, 'r') as f:
        for ele in f.readlines():
            eles = ele.strip().split("\t")
            if len(eles) < 2:
                continue
            rel2id[eles[0]] = int(eles[1])
            id2rel[int(eles[1])] = eles[0]
        
    args.nentity = nentity
    args.nrelation = nrelation
    
    adj_list, edges_y, edges_p = read_triples([os.path.join(args.data_path, "train")], args.nrelation, args.data_path, ent2id, rel2id)
    
    qa_file = args.qa_file
    question = pickle.load(open(qa_file, "rb"))
        
    new_question = []
    for ele in question:
        try:
            q = ele[0]
            head = ele[1]
            if head in freebase_id2name:
                head = freebase_id2name[head]
            path = ele[2]
            
            NotIn = False
            for r_ in path:
                if r_ not in rel2id or r_ not in relation2description:
                    NotIn = True
            if NotIn:
                continue
                     
            answers = ele[3]
            
            true_ans = set()
            for e_ in answers:
                if e_ in freebase_id2name:
                    true_ans.add(freebase_id2name[e_])
                
            if len(true_ans) == 0:
                continue
            
            new_question.append([q, list(true_ans), path, head])     
        except Exception:
            continue   
    question = new_question
    
    model = KGReasoning(args, device, adj_list)
    log_map = pickle.load(open(result_file, "rb"))
    if "What is the name of Stewie Giffin's parent that Seth Macfarlane voices" in log_map:
        del log_map["What is the name of Stewie Giffin's parent that Seth Macfarlane voices"]
    try:
        for ele in question:
            answer_id_set = set()
            answer_txt_set = set()
            q = ele[0]
            head = ele[3]
            path = ele[2]
            
            # if already answered before
            if q in log_map:
                continue
            
            rel = path[0]
            if head not in ent2id:
                continue
            answer_id_set.add(ent2id[head])

            answer_txt_set.add(head)
            
            bsz = 1        
            embedding = torch.zeros(bsz, args.nentity).to(torch.float).to(device)
            index_tensor = torch.tensor(list(answer_id_set)).unsqueeze(0).to(device)
            embedding.scatter_(-1, index_tensor, 1) # Tensor.scatter_(dim, index, src, reduce=None) 
            
            log_map[q] = []
            enumerate_idx = 0
            for rel in path:
                enumerate_idx += 1
                
                response_chatGPT = answers_by_chatGPT(answer_txt_set, rel, relation2description)
                if "[]" == response_chatGPT:
                    print("error")
                    last_results = {}
                    last_results_cnt = {}
                    
                else:
                    last_results = response_chatGPT[0][1:-1]
                    last_results_cnt = response_chatGPT[1]
                    last_results = [v for v in last_results_cnt.values()]
                    last_results.sort(key=lambda x: x[1], reverse = True)
                    
                    log_map[q].append(["chatGPT", rel, last_results])
                            
                # answer_set to embedding
                new_embedding = answer_by_kg(embedding, rel2id[rel], model)
                
                # get all the answers that have score great than a threshold   
                
                if enumerate_idx == len(path):
                    order = torch.argsort(new_embedding, dim=-1, descending=True) # Returns the indices that sort a tensor along a given dimension in ascending order by value.
                    ranking = torch.argsort(order)
                    answer_idx = order.tolist()[0][0:20]
                else:         
                    b = new_embedding > 0.0
                    __indices = b.nonzero()
                    # convert index to txt
                    __indices = __indices.tolist()
                    answer_idx = [v[-1] for v in __indices]
                answer_dict = {id2ent[v]: new_embedding[0, v].item() for v in answer_idx}
                answer_dict_lower = {id2ent[v].lower(): new_embedding[0, v].item() for v in answer_idx}
                log_map[q].append(["kg", rel, answer_dict.copy()])
                
                for name, score in last_results_cnt.items():
                    name = name[1:-1].lower() # delete "" from name
                    if name in lowerEnt2originalEnt and name not in answer_dict_lower:
                        answer_dict[lowerEnt2originalEnt[name]] = score[1] * 0.1 # 0.9 or 0.1
                
                log_map[q].append(["merge", rel, answer_dict.copy()])
                
                answer_list = list(answer_dict.items())
                answer_list.sort(key = lambda x: x[1], reverse = True)
                final_answer = []
                
                K = 10
                if enumerate_idx == len(path):
                    if answer_list:
                        pmax = answer_list[0][1]
                        for name, score in answer_list:
                            final_answer.append((name, score))
                            if len(final_answer) >= K:
                                break
                else:
                    if answer_list:
                        pmax = answer_list[0][1]
                        for name, score in answer_list:
                            if score >= pmax * 0.3:
                                final_answer.append((name, score))
                                if len(final_answer) >= K:
                                    break
                            else:
                                break
                
                log_map[q].append(['final', rel, final_answer])
                
                for name, score in final_answer:
                    new_embedding[0, ent2id[name]] = score
                    
                embedding = new_embedding
                answer_txt_set = set(['"' + name + '"' for name, score in final_answer])
            
    except Exception:
        with open(result_file, 'wb') as handle:
            pickle.dump(log_map, handle)
        traceback.print_exc()
    
    with open(result_file, 'wb') as handle:
            pickle.dump(log_map, handle)
        
        

# calculate accuracy 
def calculate_accuracy(args):
    # check the accuracy of res
    result_file = args.result_file
    
    
    qa_file = args.qa_file
    question = pickle.load(open(qa_file, "rb"))
    freebase_id2name = pickle.load(open(args.freebase, "rb"))
        
    new_question = {}
    for ele in question:
        try:
            q = ele[0]
            head = ele[1]
            if head in freebase_id2name:
                head = freebase_id2name[head]
            path = ele[2]
            answers = ele[3]
            
            true_ans = set()
            for e_ in answers:
                if e_ in freebase_id2name:
                    true_ans.add(freebase_id2name[e_])
                
            if len(true_ans) == 0:
                continue
            
            new_question[q] = list(true_ans)  
            
        except Exception:
            continue   
    question = new_question
    

    acc = 0
    average_length_of_chatGPT = []
    hist10 = []
    for i in range(10):
        hist10.append(0)
    
    
    denominator = 0
    # iterate result file, because not all question can be answered 
    res = pickle.load(open(result_file, "rb"))
    for q, rsp in res.items():
        true_ans = question[q]
        true_ans = set([a.lower() for a in true_ans])
        all_rsp_ = res[q][-1]
        all_rsp__ = all_rsp_[-1]
        all_rsp = [a[0].lower() for a in all_rsp__]
        
        Break_ = False
        denominator = denominator + 1
        for idx_, ans_at_i in enumerate(all_rsp):
            if Break_:
                break
            if ans_at_i in true_ans:
                if idx_ != 0:
                    aaaa = 1
                hist10[idx_] += 1
                Break_ = True
                break
    
    
    sum_before_hist10 = []
    for i in range(10):
        sum_before_hist10.append(0)
    sum_before_hist10[0] = hist10[0]
    for i in range(1, len(hist10)):
        sum_before_hist10[i] = sum_before_hist10[i-1] + hist10[i]


    print('HITS1_hard ' + str(sum_before_hist10[0] / denominator))
    print('HITS3_hard ' + str(sum_before_hist10[2] / denominator))
    print('HITS10_hard ' + str(sum_before_hist10[9] / denominator))

    
if __name__ == '__main__':
    args = parse_args()
    args.result_file = './chatGPT_CWQ/chatGPT_CWQ_3hop_0.1.pkl'
    args.qa_file = "./chatGPT_CWQ/CWQ_good_3hop_test.pickle"
    args.freebase = "./chatGPT_CWQ/freebase_id2name.pickle"
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    args.kbc_path = "kbc/CWQ_half/best_valid.model"
    args.data_path = "kbc/src/src_data/CWQ_half"
    args.data_path_id_information = "kbc/src/data/CWQ_half"    
    
    main(args)
    calculate_accuracy(args)

