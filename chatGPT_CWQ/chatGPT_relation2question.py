
import time
import pickle 
import traceback
import random
import openai
import json

openai.api_key = '' 

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1, # this is the degree of randomness of the model's output
        )
    return response.choices[0].message["content"]


def relation2question():
    file_name = "./chatGPT_CWQ/relation2entity_pair_no_id.pickle"
    with open(file_name, "rb") as input_file:
        relation2entity_pair = pickle.load(input_file)
    
    res_ = pickle.load(open("./chatGPT_CWQ/chatGPT_CWQ_relation2text.pkl", "rb"))
    
    try:
        for k, v in relation2entity_pair.items():
            random.shuffle(v)
            v = v[0:30]
            v = str(v)
            prompt = "Given entity pairs " + v + "\nThe relationship between the first entity and the second entity in these pairs is " + k + ". Please rewrite the relationship to a single question of entity1, so that the answer is entity2. entity1 should be able to be replaced by any first entity in these pairs. Output in Json format with key 'answer'."
            if k in res_:
                print(k)
                print(res_[k])
                continue
            response = get_completion(prompt)
            res_[k] = response
            time.sleep(0.0300)
    except Exception:
        # store res
        with open('./chatGPT_CWQ/chatGPT_CWQ_relation2text.pkl', 'wb') as handle:
            pickle.dump(res_, handle)
        
        traceback.print_exc()
    with open('./chatGPT_CWQ/chatGPT_CWQ_relation2text.pkl', 'wb') as handle:
        pickle.dump(res_, handle)
        
relation2question()
        
######
#relation2question()

