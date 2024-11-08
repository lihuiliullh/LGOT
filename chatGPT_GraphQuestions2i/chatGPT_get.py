import openai
import os
import time
import pickle 
import traceback
import json

a = 0

openai.api_key = 's' 

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1, # this is the degree of randomness of the model's output
        )

    # Extract and print the answers    
    return response.choices[0].message["content"]


prompt_example = """
I will give you a question. Please output at most 10 answers for this question in a json format. The output json has only key "answer entity".

Example1:
Q: movie interstellar starred by who?
A: {"answer entity": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain", "Bill Irwin", "Ellen Burstyn", "Michael Caine"]}

Q: Which movie starred Tom Hanks?
A: {"answer entity": ["Forrest Gump", "A Man Called Otto", "Cast Away", "The Green Mile", "The Terminal, Saving Private Ryan"]}

Q: who stars movie Titanic or Interstellar?
A: {"answer entity": ["Leonardo DiCaprio", "Kate Winslet", "Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"]}

Q: """


question = pickle.load(open("./chatGPT_GraphQuestions2i/GraphQuestions_good_2intersect_test.pkl", "rb"))
freebase_id2name = pickle.load(open("./chatGPT_GraphQuestions/freebase_id2name_graphQuestions.pickle", "rb"))

if os.path.exists("./chatGPT_GraphQuestions2i/example_chatGPT_GraphQuestions_2i_test_gpt_result.pkl"):
    with open("./chatGPT_GraphQuestions2i/example_chatGPT_GraphQuestions_2i_test_gpt_result.pkl", "rb") as input_file:
        results = pickle.load(input_file)
else:
    results = {}

try:
    for ele in question:

        q = ele[0]
        if q in results:
            continue
        prompt = prompt_example + q + '\nA:'
        
        candidate_entities = {}
        print(prompt)
        response = get_completion(prompt)
        results[q] = response
        time.sleep(0.0100)
        
except Exception:
    with open('./chatGPT_GraphQuestions2i/example_chatGPT_GraphQuestions_2i_test_gpt_result.pkl', 'wb') as handle:
        pickle.dump(results, handle)
    traceback.print_exc()


with open('./chatGPT_GraphQuestions2i/example_chatGPT_GraphQuestions_2i_test_gpt_result.pkl', 'wb') as handle:
    pickle.dump(results, handle)
# print("Completion for Text 1:")
# print(response)
