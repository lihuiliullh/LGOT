import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from kg_model import KGReasoning
import pickle
from collections import defaultdict
from tqdm import tqdm
from relation2description import *
import traceback
import io
from utils import *
from dataset import TestDataset
import openai
from ast import literal_eval

# code is slightly different from 2u. In prompt and other things

openai.api_key = 'sk' 

def get_answer_chatGPT(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1, # this is the degree of randomness of the model's output
        )
    return response.choices[0].message["content"]


relation2description_metaQA = {
    "release_year": "text_a was released in which year?",
    "release_year_reverse": "which movies were released in year text_a?", 
    "has_genre": "movie text_a has which genre?",
    "has_genre_reverse": "which films belong to genre text_a?",
    "has_tags": "text_a has tags what?",
    "has_tags_reverse": "tag text_a has which film?",
    "starred_actors": "movie text_a has actors?",
    "starred_actors_reverse": "text_a starred in which movies?",
    "directed_by": "movie text_a was directed by who?",
    "directed_by_reverse": "text_a directed which movies?",
    "written_by": "film text_a was written by who?",
    "written_by_reverse": "text_a wrote which movies?",
    "in_language": "text_a was spoken in which language?",
    "in_language_reverse": "language text_a was spoken in which movies?",
    "has_imdb_rating": "movie text_a has what imdb rating?",
    "has_imdb_rating_reverse": "which movies have imdb rating text_a?",
    "has_imdb_votes": "movie text_a has what imdb votes?",
    "has_imdb_votes_reverse": "which movies have imdb votes text_a?",
    "noop": "output itself",
    "noop_reverse": "output itself"
}


all_entity_set = set()
prompt_example = """
I will give you a question. Please output at most 10 answers for this question in a string format separated by commas. Do not give multiple answers.

Examples:
Q: who starred movie interstellar but not starred movie Titanic?
A: "Matthew McConaughey", "Anne Hathaway", "Jessica Chastain", "Bill Irwin", "Ellen Burstyn", "Michael Caine"

Q: Which movie starred Tom Hanks?
A: "Forrest Gump", "A Man Called Otto", "Cast Away", "The Green Mile", "The Terminal", "Saving Private Ryan"

Q: who stars movie Titanic or Interstellar but not stars The Snow White?
A: "Leonardo DiCaprio", "Kate Winslet", "Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"

Q: """

def extract_answers(s):
    i = s.find('"')
    ans = set()
    while i != -1:
        i += 1
        j = s.find('"', i)
        if j == -1:
            break
        ans.add(s[i : j])
        i = s.find('"', j + 1)
    return ans


def answers_by_chatGPT(head_set, rel, relation2description=relation2description_metaQA):
    prompt = relation2description[rel]
        
    candidate_entities = list(head_set)
    if len(candidate_entities) <= 2:
        ents = " or ".join(candidate_entities)
    else:
        ents = ", ".join(candidate_entities[: -1]) + ", or " + candidate_entities[-1]
    
    prompt = prompt.replace("text_a", ents)
    
    prompt = prompt_example + prompt + '\nA:'
    
    response = get_answer_chatGPT(prompt)
    response = extract_answers(response)
    return response


    
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


def parse_query(args, kg_model, query_structure, query_element_id, idx=0):
    all_relation_flag = True
    exec_query = []
    exec_query_LLM = []
    LLM_ans = set()
    
    for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
        if ele not in ['r', 'n']: # 
            all_relation_flag = False
            break
    # all_relation_flag means projection or negation at last step
    # how many steps executed in the computation graph
    if all_relation_flag:
        if query_structure[0] == 'e':
            # initialization
            embedding = torch.zeros(1, args.nentity).to(torch.float).to(args.device)
            embedding.scatter_(-1, query_element_id[:, idx].unsqueeze(-1), 1) # Tensor.scatter_(dim, index, src, reduce=None)
            for _v in query_element_id[:, idx].tolist():
                _v = id2ent[_v]
                LLM_ans.add(_v)
            
            exec_query.append(query_element_id[:, idx].item())
            exec_query_LLM.append(query_element_id[:, idx].item())
            idx += 1
        else:
            # more complex query structure, like ip, pin, 2u-DNF ... 
            embedding, LLM_ans, idx, pre_exec_query, pre_exec_query_llm = parse_query(args, kg_model, query_structure[0], query_element_id, idx)
            exec_query.append(pre_exec_query)
            exec_query_LLM.append(pre_exec_query_llm)
        
        r_exec_query = []
        r_exec_query_llm = []
        for i in range(len(query_structure[-1])): # iterate all relations in the path
            if query_structure[-1][i] == 'n':
                assert (query_element_id[:, idx] == -2).all()
                r_exec_query.append('n')
                r_exec_query_llm.append('n')
            else:
                r_embedding = kg_model.get_relation_embedding(query_element_id[0, idx])
                
                # call QTO
                if (i < len(query_structure[-1]) - 1) and query_structure[-1][i+1] == 'n':
                    embedding, r_argmax = kg_model.relation_projection(embedding, r_embedding, True)
                    # here should use negative
                    LLM_ans = answers_by_chatGPT(LLM_ans, id2rel[query_element_id[0, idx].item()])
                    LLM_ans = all_entity_set.difference(LLM_ans)
                else:
                    embedding, r_argmax = kg_model.relation_projection(embedding, r_embedding, False)
                    LLM_ans = answers_by_chatGPT(LLM_ans, id2rel[query_element_id[0, idx].item()])
                    
                r_exec_query.append((query_element_id[0, idx].item(), r_argmax))
                r_exec_query.append('e')
                
                r_exec_query_llm.append((query_element_id[0, idx].item(), LLM_ans))
                r_exec_query_llm.append('e')
                
            idx += 1
        r_exec_query.pop()
        r_exec_query_llm.pop()
        
        exec_query.append(r_exec_query)
        exec_query.append('e')
        
        exec_query_LLM.append(r_exec_query_llm)
        exec_query_LLM.append('e')
        
    else:
        # here is union or intersection at last step
        embedding_list = []
        LLM_ans_list = []
        union_flag = False
        for ele in query_structure[-1]:
            if ele == 'u':
                union_flag = True
                query_structure = query_structure[:-1]
                break
        for i in range(len(query_structure)):
            embedding, LLM_ans, idx, pre_exec_query, pre_exec_query_llm = parse_query(args, kg_model, query_structure[i], query_element_id, idx)
            embedding_list.append(embedding)
            LLM_ans_list.append(LLM_ans)
            exec_query.append(pre_exec_query)
            exec_query_LLM.append(pre_exec_query_llm)
        if union_flag:
            embedding = kg_model.union(torch.stack(embedding_list))
            # LLM union
            LLM_ans = set.union(*LLM_ans_list)
            idx += 1
            exec_query.append(['u'])
            exec_query_LLM.append(['u'])
        else:
            embedding = kg_model.intersection(torch.stack(embedding_list))
            # LLM intersection
            LLM_ans = set.intersection(*LLM_ans_list)
        exec_query.append('e')
        exec_query_LLM.append('e')
    
    return embedding, LLM_ans, idx, exec_query, exec_query_LLM


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def flatten_query(queries):
    all_queries = []
    for query_structure in queries: # query_structure is the key
        a__ = queries[query_structure] # type is class 'set'
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


def main(args, test=False):
    set_global_seed(args.seed)
    device = args.device
    
    query_dir = args.query_dir
        
    args.fraction = 10 # same as QTO
    args.thrshd = 0.001 # same as QTO
    
    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    
    global id2ent, id2rel
    with open('%s/id2ent.pkl'%args.data_path, 'rb') as f:
        id2ent = pickle.load(f)
    with open('%s/ent2id.pkl'%args.data_path, 'rb') as f:
        ent2id = pickle.load(f)
    
    global all_entity_set
    for v in ent2id:
        all_entity_set.add(v)
    
    
    with open('%s/id2rel.pkl'%args.data_path, 'rb') as f:
        id2rel = pickle.load(f)
    
    rel2id = {}
    for k, v in id2rel.items():
        rel2id[v] = k
    
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    ######
    if test:
        calculate_accuracy(args, id2ent)
        return
    
    
    adj_list, edges_y, edges_p = read_triples_QTO([os.path.join(args.data_path, "train.txt")], args.nrelation, args.data_path)
    
    logic_query_QTO = pickle.load(open(query_dir, "rb"))
    logic_query_QTO = flatten_query(logic_query_QTO)
    test_dataloader = DataLoader(
        TestDataset(
            logic_query_QTO, 
            args.nentity, 
            args.nrelation, 
        ), 
        batch_size=1,
        num_workers=1,
        collate_fn=TestDataset.collate_fn,
    )
    
    result_file = args.result_file
    final_results = {}
    
    if os.path.exists(result_file):
        final_results = pickle.load(open(result_file, "rb"))
    else:
        final_results = {}
    
    model = KGReasoning(args, device, adj_list)
    
    try:
        for queries, queries_unflatten, query_structures in tqdm(test_dataloader):  
            if str(queries_unflatten) in final_results:
                continue
            query_structure = query_structures[0]
            query_element_id = torch.tensor(queries).to(device)
            
            embedding, LLM_ans, idx, exec_query, exec_query_LLM = parse_query(args, model, query_structure, query_element_id, idx=0)
            print(query_structure)
            print(LLM_ans)
            a = 0
            final_results[str(queries_unflatten)] = (LLM_ans, embedding)
    # save results
    except Exception:
        with open(result_file, 'wb') as handle:
            pickle.dump(final_results, handle)
        traceback.print_exc()
        return
    
    with open(result_file, 'wb') as handle:
        pickle.dump(final_results, handle)


def calculate_accuracy(args, id2ent):
    ent2id ={}
    for k, v in id2ent.items():
        ent2id[v] = k
    
    result_file = args.result_file
    with open(result_file, 'rb') as f:
        if torch.cuda.is_available():
            logic_query_QTO = pickle.load(f)
        else:
            logic_query_QTO = CPU_Unpickler(f).load()
            
    hard_ans = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    hard_ans_new = {}
    for k, v in hard_ans.items():
        new_v = [id2ent[_] for _ in v]
        hard_ans_new[str(k)] = set(new_v)
    hard_ans = hard_ans_new
    
    denominator = 0
    acc = 0
    average_length_of_chatGPT = []
    hist10 = []
    for i in range(10):
        hist10.append(0)
    
    for k, v in logic_query_QTO.items():
        k = k[1:-1]
        k_id = literal_eval(k)
        id_1 = k_id[0][0]
        id_2 = k_id[1][0]
        denominator = denominator + 1
        llms_ans = v[0]
        candidate_by_QTO = v[1]
        
        true_answers = hard_ans[k]
        
        for a_ in llms_ans:
            a_ = a_.strip()
            if a_ not in ent2id:
                continue
            
            id = ent2id[a_]
            if int(id) == int(id_1) or int(id) == (id_2):
                continue
            candidate_by_QTO[0][id] += 0.1
            
        
        order = torch.argsort(candidate_by_QTO, dim=-1, descending=True) # Returns the indices that sort a tensor along a given dimension in ascending order by value.
        ranking = torch.argsort(order)
        
        _idx = order[0][0:10].tolist()
        idx_ = 0
        for id in _idx:
            id = id2ent[id]
            if id in true_answers:
                hist10[idx_] += 1
                break
            idx_ += 1
    
    sum_before_hist10 = []
    for i in range(10):
        sum_before_hist10.append(0)
    sum_before_hist10[0] = hist10[0]
    for i in range(1, 10):
        sum_before_hist10[i] = sum_before_hist10[i-1] + hist10[i]


    print('HITS1_hard ' + str(sum_before_hist10[0] / denominator))
    print('HITS3_hard ' + str(sum_before_hist10[2] / denominator))
    print('HITS10_hard ' + str(sum_before_hist10[9] / denominator))
    


if __name__ == '__main__':
    args = parse_args()
    args.kbc_path = "kbc/metaQA_half/best_valid.model"
    args.data_path = "data/metaQA_pin_half"
    args.data_path_id_information = "kbc/src/data/metaQA_half"
    query_dir = 'data/metaQA_pin_half/test-queries.pkl'
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.result_file = "./LGOT/" + args.data_path.split("/")[1] + "_combine_res.pkl"
    
    main(args, test=True)

