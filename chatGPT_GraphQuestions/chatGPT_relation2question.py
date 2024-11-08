
import time
import pickle 
import traceback
import random
import openai
import json
import os

openai.api_key = '' 

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1, # this is the degree of randomness of the model's output
        )
    return response.choices[0].message["content"]


def pair2string(v):
    a = 0
    for idx in range(len(v)):
        v[idx] = ('[' + v[idx][0] + ']', '[' + v[idx][1] + ']')
    return str(v)


def relation2question():
    file_name = "./chatGPT_GraphQuestions/relation2entity_pair_no_id_GraphQuestions.pickle"
    with open(file_name, "rb") as input_file:
        relation2entity_pair = pickle.load(input_file)
    
    # filter out useful relations
    useful_relation_file_2hop = "./data/GraphQuestions-2hop/test-queries.pkl"
    useful_relation_file_2hop = pickle.load(open(useful_relation_file_2hop, "rb"))
    useful_relation_file_2hop = useful_relation_file_2hop[('e', ('r', 'r'))] 
    
    # read relation dict
    relation_dict_file = "./data/GraphQuestions-2hop/id2rel.pkl"
    relation_dict_file = pickle.load(open(relation_dict_file, "rb"))
    
    useful_relation_set = set()
    for value in useful_relation_file_2hop:
        useful_relation_set.add(relation_dict_file[value[1][0]])
        useful_relation_set.add(relation_dict_file[value[1][1]])
    
    if os.path.exists("./chatGPT_GraphQuestions/chatGPT_GraphQuestions_relation2text.pkl"):
        with open("./chatGPT_GraphQuestions/chatGPT_GraphQuestions_relation2text.pkl", "rb") as input_file:
            res_ = pickle.load(input_file)
    else:
        res_ = {}
    #res_ = pickle.load(open("./chatGPT_GraphQuestions/chatGPT_GraphQuestions_relation2text.pkl", "rb"))
    
    try:
        for k, v in relation2entity_pair.items():
            if k in res_:
               continue
            if k not in useful_relation_set:
                continue
            random.shuffle(v)
            v = v[0:30]
            v = pair2string(v)
            prompt = "Given entity pairs " + v + "\nThe relationship between the first entity and the second entity in these pairs is " + k + ". Please rewrite the relationship to a single question of entity1, so that the answer is entity2. entity1 should be able to be replaced by any first entity in these pairs. Output in Json format with key 'answer'."
            response = get_completion(prompt)
            res_[k] = response
            time.sleep(0.0300)
    except Exception:
        # store res
        with open('./chatGPT_GraphQuestions/chatGPT_GraphQuestions_relation2text.pkl', 'wb') as handle:
            pickle.dump(res_, handle)
        
        traceback.print_exc()
    with open('./chatGPT_GraphQuestions/chatGPT_GraphQuestions_relation2text.pkl', 'wb') as handle:
        pickle.dump(res_, handle)
        
relation2question()
        
######
#relation2question()

