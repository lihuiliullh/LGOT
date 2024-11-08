import openai
import os
import pickle
import time
import traceback
import json


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

prompt_example = """
I will give you a question. Please output as many answers as possible, but at most 10 answers for this question in a string format separated by commas. Do not give multiple answers.

Examples:
Q: movie interstellar starred by who or who directed movie Forrest Gump?
A: "Matthew McConaughey", "Anne Hathaway", "Jessica Chastain", "Bill Irwin", "Ellen Burstyn", "Michael Caine"

Q: Which movie starred Tom Hanks or which movie directed by Steven Spielberg?
A: "Forrest Gump", "A Man Called Otto", "Cast Away", "The Green Mile", "The Terminal", "Saving Private Ryan"

Q: who stars movie Titanic or or who directed movie Interstellar?
A: "Leonardo DiCaprio", "Kate Winslet", "Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"

Q: """


openai.api_key = '' 

def get_answer_chatGPT(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1, # this is the degree of randomness of the model's output
        )
    return response.choices[0].message["content"]


def main():
    question = []
    with open("./chatGPT_metaQA_2u/metaQA_2u.txt", 'r') as f:
        for line in f.readlines():
            ele = line.strip().split("\t")
            if len(ele) < 2:
                continue
            question.append(ele[0])
    f.close()
    
    
    
    if os.path.exists("./chatGPT_metaQA_2u/chatGPT_metaQA_2u_test_gpt.pkl"):
        res = pickle.load(open("./chatGPT_metaQA_2u/chatGPT_metaQA_2u_test_gpt.pkl", "rb"))
    else:
        res = {}
    
    # iterate question and ask chatGPT to answer the question
    try:
        for q in question:
            if q in res:
                continue
            
            prompt = prompt_example + q + "\nA:"
            response = get_answer_chatGPT(prompt)
            res[q] = response
            time.sleep(0.0300)
    except Exception:
        # store res
        with open('./chatGPT_metaQA_2u/chatGPT_metaQA_2u_test_gpt.pkl', 'wb') as handle:
            pickle.dump(res, handle)
        
        traceback.print_exc()
    
    with open('./chatGPT_metaQA_2u/chatGPT_metaQA_2u_test_gpt.pkl', 'wb') as handle:
        pickle.dump(res, handle)


def accuracy():
    
    question = {}
    question_rel = {}
    with open("./chatGPT_metaQA_2u/metaQA_2u.txt", 'r') as f:
        for line in f.readlines():
            ele = line.strip().split("\t")
            if len(ele) < 2:
                continue
            q = ele[0]
            ans = ele[1].split("|")
            question[q] = set(ans)
            question_rel[q] = ele[2]
    f.close()

    res = pickle.load(open("./chatGPT_metaQA_2u/chatGPT_metaQA_2u_test_gpt.pkl", "rb"))

    for k, v in list(res.items()):
        if k not in question:
            del res[k]

    # check the accuracy of res
    acc = 0
    average_length_of_chatGPT = []
    hist10 = []
    for i in range(10):
        hist10.append(0)

    dedominator = 0
    wrong_parse = 0
    for q, rsp in res.items():
        rell = question_rel[q]
        if 'tag' in rell:
            wrong_parse = wrong_parse + 1
            continue
        try:
            all_rsp = extract_answers(rsp)
        except:
            wrong_parse = wrong_parse + 1
            continue
        all_rsp = [str(a) for a in all_rsp]
        all_rsp = [a.lower() for a in all_rsp]
        true_ans = question[q]
        
        Break_ = False
        dedominator = dedominator + 1
        for idx_, ans_at_i in enumerate(all_rsp):
            if Break_:
                break
            for aa in true_ans:
                aa = aa.lower()
                if (aa in ans_at_i or ans_at_i in aa) and idx_ < 10:
                    hist10[idx_] += 1
                    Break_ = True
                    break
            
            # if not Break_ and  idx_ < 10:
            #     print(ans_at_i)
            #     if ans_at_i == ' arnaud desplechin':
            #         a = 0

    sum_before_hist10 = []
    for i in range(10):
        sum_before_hist10.append(0)
    sum_before_hist10[0] = hist10[0]
    for i in range(1, len(hist10)):
        sum_before_hist10[i] = sum_before_hist10[i-1] + hist10[i]


    print('HITS1_hard ' + str(sum_before_hist10[0] / dedominator))
    print('HITS3_hard ' + str(sum_before_hist10[2] / dedominator))
    print('HITS10_hard ' + str(sum_before_hist10[9] / dedominator))


main()
accuracy()