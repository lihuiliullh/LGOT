import openai
import os
import time
import pickle 
import traceback
import json
from collections.abc import Mapping

a = 0


openai.api_key = 's' 

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1, # this is the degree of randomness of the model's output
        )
    return response.choices[0].message["content"]
  

POTENTIAL_ANS_1 = """
[
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "Telescope type"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "Reflecting telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "Hale Telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "48-inch Schmidt telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "60-inch telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "200-inch Hale Telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "Schmidt telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "Reflecting Schmidt telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "48-inch reflecting telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "60-inch reflecting telescope"
  }
]
"""

FUNCK2 = """
[
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "Telescope type"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "Reflecting telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "Hale Telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "48-inch Schmidt telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "60-inch telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "200-inch Hale Telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "Samuel Oschin Telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "48-inch Oschin Schmidt Telescope"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "150-foot Solar Tower"
  },
  {
    "input entity": "Mount Palomar Observatory",
    "answer entity": "18-inch Schmidt telescope"
  }
]
"""


question = pickle.load(open("./chatGPT_GraphQuestions2i/GraphQuestions_good_2intersect_test.pkl", "rb"))
freebase_id2name = pickle.load(open("./chatGPT_GraphQuestions/freebase_id2name_graphQuestions.pickle", "rb"))
    
# load the previous step
last_results_map2 = pickle.load(open("chatGPT_GraphQuestions2i/GraphQuestions_good_2intersect_test_relation2.pkl", "rb"))
last_results_map1 = pickle.load(open("chatGPT_GraphQuestions2i/GraphQuestions_good_2intersect_test_relation1.pkl", "rb"))

evidence_2i = pickle.load(open("./chatGPT_GraphQuestions2i/evidence_2i_GraphQuestions.pkl", 'rb'))

candidate_entities_map = {}

try:
    for ele in question:
        q = ele[0]
        
        head = ele[1][0][1] # should map id to name
        head1_id = ele[1][0][0]
        head2_id = ele[1][1][0]
        rel1 = ele[1][3]
        rel2 = ele[1][4]
        ans = ele[1][2]
        
        if q not in last_results_map1:
            continue
        
        candidate_entities = []
        
        last_results1 = last_results_map1[q]
        last_results2 = last_results_map2[q]
        
        try:
            xxxx = last_results1
            
            last_results1 = json.loads(last_results1)
            if 'top10 answers' in last_results1:
              top10 = last_results1['top10 answers']
              last_results1 = top10
            else:
              last_results1 = last_results1['answer entity']
            
              
            
            try:
                last_results2 = json.loads(last_results2)
                if 'top10 answers' in last_results2:
                  last_results2 = last_results2['top10 answers']
                else:
                  last_results2 = last_results2['answer entity']
            except:
                last_results2 = last_results2.replace("\n", "")
                last_results2 = last_results2.replace("{", "")
                last_results2 = last_results2.replace("}", "")
                last_results2 = last_results2.replace(",", "")
                last_results2 = last_results2.replace('"', "")
                last_results2 = last_results2.split("answer entity")
                last_results2 = last_results2[1:]
                last_results2 = [xx.split("input entity")[0] for xx in last_results2]
                last_results2 = [a.strip() for a in last_results2]
                a = 0
        except:
            wrong = 1

        # calculate intersection
        # since the output may be different, we simply use word intersection
        goodness_idx = [0 for a in last_results1]
        if isinstance(last_results1, str):
            last_results1 = [last_results1]
        if isinstance(last_results2, str):
            last_results2 = [last_results2]
            
        new_last_results1 = [set(a.split(" ")) for a in last_results1]
        new_last_results2 = [set(a.split(" ")) for a in last_results2]
        for xxx in new_last_results2:
            for in_idx, ans_ele in enumerate(new_last_results1):
                tmp = xxx.intersection(ans_ele)
                if len(tmp) >= len(ans_ele) * 0.5 or len(tmp) >= len(xxx) * 0.5:
                    goodness_idx[in_idx] = 1
        a = 0
        
        for cur_idx, idx_ele in enumerate(goodness_idx):
            if idx_ele == 1:
                candidate_entities.append(last_results1[cur_idx])
        
        existing_set = set(candidate_entities)
        
        # reorder the answers, first intersction, then left of 1, then left of 2
        if len(candidate_entities) == 0:
          fu = 0
        for potential_asn in last_results1:
          if potential_asn not in existing_set:
            candidate_entities.append(potential_asn)
        for potential_asn in last_results2:
          if potential_asn not in existing_set:
            candidate_entities.append(potential_asn)
            
        ########add
        # add results for logic reasoning 
        logic_query_res = evidence_2i[(head1_id, head2_id, rel1, rel2)]
        for ee_ in logic_query_res:
            if ee_ in freebase_id2name:
                candidate_entities.insert(0, freebase_id2name[ee_])
            else:
                candidate_entities.insert(0, ee_)
                
        candidate_entities_map[q] = candidate_entities
        
except Exception:
    traceback.print_exc()
    
##################################################################
if os.path.exists("./chatGPT_GraphQuestions2i/chatGPT_GraphQuestions_2i_test_evaluate.pkl"):
    res = pickle.load(open("./chatGPT_GraphQuestions2i/chatGPT_GraphQuestions_2i_test_evaluate.pkl", "rb"))
else:
    res = {}


merged_res = candidate_entities_map
acc = 0
average_length_of_chatGPT = []
hist10 = []
for i in range(10):
    hist10.append(0)

try:

  for ele in question:
      true_ans = set()
      for c_ans in ele[1][2]:
          true_ans.add(c_ans['entity_name'])
          
      q = ele[0]
      
      if q in res:
        continue
      
      if q not in merged_res:
          continue
      answer_of_gpt = merged_res[q]
      all_rsp = [str(v).lower() for v in answer_of_gpt]
      
      true_ans_ = []
      for e__ in true_ans:
          if e__ in freebase_id2name:
              e__ = freebase_id2name[e__]
          true_ans_.append(e__)
      true_ans = [str(v).lower() for v in true_ans_]
      
      ######################################################
      prompt = ""
      prompt = prompt + "Replace the answers with correct answers if the answers in the choices are not correct.\n"
      prompt = prompt + "Given the question: " + q + "\n" + "And its potential answer choices:\n"
      for p_idx, potential_ans in enumerate(all_rsp):
          in_txt = "Choice " + str(p_idx) + " : " + potential_ans + "\n"
          prompt = prompt + in_txt
      prompt = prompt + 'output the top10 answers in a json format before 2021 in 100 words.\nthe output json has key "input", "answer"\n'
      prompt = prompt + "The answers don't need to be in the potential answer choices."
      #print(prompt)
      response = get_completion(prompt)
      print(q)
      res[q] = response
      time.sleep(0.0300)
except Exception:
    # store res
    with open('./chatGPT_GraphQuestions2i/chatGPT_GraphQuestions_2i_test_evaluate.pkl', 'wb') as handle:
        pickle.dump(res, handle)
    
    traceback.print_exc()

with open('./chatGPT_GraphQuestions2i/chatGPT_GraphQuestions_2i_test_evaluate.pkl', 'wb') as handle:
    pickle.dump(res, handle)
    



        