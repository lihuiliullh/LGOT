import openai
import os
import time
import pickle 
import traceback
import json
from collections.abc import Mapping

a = 0



question = pickle.load(open("./chatGPT_GraphQuestions2i/GraphQuestions_good_2intersect_test.pkl", "rb"))
freebase_id2name = pickle.load(open("./chatGPT_GraphQuestions/freebase_id2name_graphQuestions.pickle", "rb"))
    
# load the previous step
last_results_map2 = pickle.load(open("chatGPT_GraphQuestions2i/example_GraphQuestions_good_2intersect_test_relation2.pkl", "rb"))
last_results_map1 = pickle.load(open("chatGPT_GraphQuestions2i/example_GraphQuestions_good_2intersect_test_relation1.pkl", "rb"))


last_results_map2 = pickle.load(open("chatGPT_GraphQuestions2i/example_chatGPT_GraphQuestions2i_test_evaluate_origin.pkl", "rb"))
last_results_map1 = pickle.load(open("chatGPT_GraphQuestions2i/example_chatGPT_GraphQuestions2i_test_evaluate_origin.pkl", "rb"))


evidence_2i = pickle.load(open("./chatGPT_GraphQuestions2i/evidence_2i_GraphQuestions.pkl", 'rb'))
false_evidence_2i = pickle.load(open("./chatGPT_GraphQuestions2i/false_evidence_2i_GraphQuestions.pkl", 'rb'))

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
                
        last_results1 = json.loads(last_results1)
        last_results1 = last_results1['answer entity']
        
        try:
          last_results2 = json.loads(last_results2)
        except:
          print(last_results2)
        last_results2 = last_results2['answer entity']
        
        # calculate intersection
        # since the output may be different, we simply use word intersection
        last_results1 = [a.lower() for a in last_results1]
        last_results2 = [a.lower() for a in last_results2]
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
              
        if len(candidate_entities) < 10:
            false_cand = false_evidence_2i[(head1_id, head2_id, rel1, rel2)]
            false_cand = list(false_cand)[0: 10 - len(candidate_entities)]
            for x_ in false_cand:
              if x_ in freebase_id2name:
                candidate_entities.append(freebase_id2name[x_])
              else:
                candidate_entities.append(x_)
                
        candidate_entities_map[q] = candidate_entities
        
except Exception:
    traceback.print_exc()
    
##################################################################
merged_res = candidate_entities_map
acc = 0
average_length_of_chatGPT = []
hist10 = []
for i in range(10):
    hist10.append(0)

denominator = 0

for ele in question:
    true_ans = set()
    for c_ans in ele[1][2]:
        true_ans.add(c_ans['entity_name'])
        
    q = ele[0]
    
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
    Break_ = False
    denominator = denominator + 1
    for idx_, ans_at_i in enumerate(all_rsp):
        if Break_:
            break
        for aa in true_ans:
            aa = aa.lower()
            if (aa in ans_at_i or ans_at_i in aa) and idx_ < 10:
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
print(denominator)
