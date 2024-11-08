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


relation2description = pickle.load(open("./chatGPT_GraphQuestions2i/chatGPT_GraphQuestions_relation2text.pkl", "rb"))
for k, v in relation2description.items():
    v = json.loads(v)['answer']
    if 'entity1' not in v:
        if '[' not in v:
            v = v[0:-1] + " entity1" + "?"
        else:
            v = v.split("[")[0] + 'entity1' + v.split("[")[1].split("]")[1]
    relation2description[k] = v
# output the top10 answers for the the question "Tom Hanks or Tom Cruise starred in which movies?" in a json format before 2021 in 100 words. 
# the output json has key "input entity", "answer entity"


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

if os.path.exists("./chatGPT_GraphQuestions2i/example_GraphQuestions_good_2intersect_test_relation1.pkl"):
    with open("./chatGPT_GraphQuestions2i/example_GraphQuestions_good_2intersect_test_relation1.pkl", "rb") as input_file:
        results = pickle.load(input_file)
else:
    results = {}
    

try:
    for ele in question:
        q = ele[0]
        
        head = ele[1][0][1] # should map id to name
        rel1 = ele[1][3]
        rel2 = ele[1][4]
        
        if q in results:
            continue
        
        candidate_entities = {head}
        # ('comic_books.comic_book_series', 'Comic Book Series'), [], 'comic_books.comic_bo...ntinued_by', 'comic_books.comic_bo...ies.issues')
        rel = rel1
        if rel not in relation2description:
            wrong_here = 1
            continue
        prompt = relation2description[rel]
        
        ents = " or ".join(list(candidate_entities))
        prompt = prompt.replace("entity1", ents)
        
        prompt = prompt_example + prompt + '\nA:'
        
        candidate_entities = {}
        print(prompt)
        response = get_completion(prompt)
        results[q] = response
        time.sleep(0.0100)
        
except Exception:
    with open('./chatGPT_GraphQuestions2i/example_GraphQuestions_good_2intersect_test_relation1.pkl', 'wb') as handle:
        pickle.dump(results, handle)
    
    traceback.print_exc()
    

with open('./chatGPT_GraphQuestions2i/example_GraphQuestions_good_2intersect_test_relation1.pkl', 'wb') as handle:
    pickle.dump(results, handle)
# print("Completion for Text 1:")
# print(response)