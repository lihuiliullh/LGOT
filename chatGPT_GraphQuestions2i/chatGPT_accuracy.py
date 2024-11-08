import openai
import os
import time
import pickle 
import traceback
import statistics
import json
from collections.abc import Mapping


question = pickle.load(open("./chatGPT_GraphQuestions2i/GraphQuestions_good_2intersect_test.pkl", "rb"))
freebase_id2name = pickle.load(open("./chatGPT_GraphQuestions/freebase_id2name_graphQuestions.pickle", "rb"))

last_results_map = pickle.load(open("./chatGPT_GraphQuestions2i/example_chatGPT_GraphQuestions_2i_test_gpt_result.pkl", "rb"))
last_results_map = pickle.load(open("./chatGPT_GraphQuestions2i/cot_example_chatGPT_GraphQuestions_2i_test_gpt_result.pkl", "rb"))


candidate_entities_map = {}
merged_res = {}

try:
    for question_ele in question:
        q = question_ele[0]
        
        if q not in last_results_map:
            continue
        
        last_results = last_results_map[q]
        
        try:
            last_results = json.loads(last_results)
            ans_parsed = last_results["answer entity"]
            merged_res[q] = ans_parsed
        except:
            print(last_results)
            traceback.print_exc()
            continue
except Exception:
    traceback.print_exc()


###################################
###################################
evidence_2i = pickle.load(open("./chatGPT_GraphQuestions2i/evidence_2i_GraphQuestions.pkl", 'rb'))
false_evidence_2i = pickle.load(open("./chatGPT_GraphQuestions2i/false_evidence_2i_GraphQuestions.pkl", 'rb'))

candidate_entities_map = merged_res
merged_res = {}
for ele in question:
        q = ele[0]
        
        head = ele[1][0][1] # should map id to name
        head1_id = ele[1][0][0]
        head2_id = ele[1][1][0]
        rel1 = ele[1][3]
        rel2 = ele[1][4]
        ans = ele[1][2]
        
        final_res = []
        
        # add results for logic reasoning 
        logic_query_res = evidence_2i[(head1_id, head2_id, rel1, rel2)]
        # for ee_ in logic_query_res:
        #     if ee_ in freebase_id2name:
        #         final_res.append(freebase_id2name[ee_])
        #     else:
        #         final_res.append(ee_)

        if q not in candidate_entities_map:
            print(q)
            continue
        
        for x_ in candidate_entities_map[q]:
            final_res.append(x_)
            
        # if len(final_res) < 10:
        #     false_cand = false_evidence_2i[(head1_id, head2_id, rel1, rel2)]
        #     false_cand = list(false_cand)[0: 10 - len(final_res)]
        #     for x_ in false_cand:
        #         final_res.append(x_)
    
        merged_res[q] = final_res
# a = 0
###################################
###################################

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
    
    Break_ = False
    denominator = denominator + 1
    for idx_, ans_at_i in enumerate(all_rsp): # all found answers
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


