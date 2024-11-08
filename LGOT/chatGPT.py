import openai
import os
import time
import pickle 
import traceback

# messages=[ 
# {"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Who won the world series in 2020?"}, {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}, 
# {"role": "user", "content": "Where was it played?"} 
# ] 

openai.api_key = 'sk-' 


# gpt-4-1106-preview
# gpt-3.5-turbo
def get_answer_chatGPT(prompt, model="gpt-3.5-turbo", n=6):
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2, # this is the degree of randomness of the model's output
        logprobs=True,
        n=n,
        )
    
    # Extract and print the answers    
    cnt = dict()
    for choice in response.choices:
        content = choice.message["content"]
        answers = []
        answer = []
        inner = False
        for char in content:
            if char == '"':
                if inner:
                    answer = ''.join(answer)
                    answers.append(f'"{answer}"')
                    inner = False
                    answer = []
                else:
                    inner = True
            else:
                if inner:
                    answer.append(char)
        #content = set(choice.message["content"].split(", "))
        for answer in answers:
            key = answer.lower()
            if key not in cnt:
                cnt[key] = [answer, 0]
            cnt[key][1] += 1
    if not cnt:
        print(content)
        return "[]"
    maxcnt = max([val[1] for val in cnt.values()])
    a_ = [answer for key, (answer, num) in cnt.items() if num >= maxcnt * 0.3]
    return ("[" + ", ".join(a_) + "]", {answer.lower(): [cnt[answer.lower()][0], cnt[answer.lower()][1] / n] for answer in a_})

