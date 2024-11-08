import random 

# create new data 2u, pin (all questions about person)
# for 2u, only use 1-hop question
# for pin, use two-hop question and 1-hop negation

# create train, test, valid according to path
def create_2u(file_path, num_points=600):
    # open file and find out all questions related to person
    good_questions = []
    with open(file_path) as f:
        for line in f.readlines():
            elements = line.strip().split("\t")
            if len(elements) < 2:
                continue
            if elements[0].lower().startswith("who"):
                good_questions.append(elements)
    
    # create dataset
    # randomly sample two questions from 
    new_data = []
    for i in range(num_points):
        first = random.randint(0, len(good_questions)-1)
        second = random.randint(0, len(good_questions)-1)
        while second == first:
            second = random.randint(0, len(good_questions)-1)

        first_question = good_questions[first]
        second_question = good_questions[second]
        
        new_question = first_question[0] + " or " + second_question[0]
        answers = first_question[1] + "|" + second_question[1]
        paths = "(" + first_question[2] + ", " + second_question[2] + ")"
        new_data.append([new_question, answers, paths])
    
    # how to store data
    # QTO will run it
    # use the orignal data format
    f = open("./metaQA_create_data/metaQA_2u.txt", "w")
    for d in new_data:
        f.write("\t".join(d) + "\n")
    f.close()        
    
    return



def create_pin(two_hop_path, one_hop_path, num_points=600):
    # read two hop questions
    two_hop_questions = []
    with open(two_hop_path) as f:
        for line in f.readlines():
            elements = line.strip().split("\t")
            if len(elements) < 2:
                continue
            if elements[0].lower().startswith("who"):
                two_hop_questions.append(elements)
    
    # read one hop questions
    one_hop_questions = []
    with open(one_hop_path) as f:
        for line in f.readlines():
            elements = line.strip().split("\t")
            if len(elements) < 2:
                continue
            if elements[0].lower().startswith("who"):
                one_hop_questions.append(elements)
    
    # two hop plus one negative
    new_data = []
    for i in range(num_points):
        first = random.randint(0, len(two_hop_questions)-1)
        second = random.randint(0, len(one_hop_questions)-1)
        
        first_question = two_hop_questions[first]
        second_question = one_hop_questions[second]
        
        new_question = first_question[0] + " but do not" + second_question[0][3:]
        # delete all second answer from the first one
        ans_set1 = set(first_question[1].split("|"))
        ans_set2 = set(second_question[1].split("|"))
        z = ans_set1.difference(ans_set2)
        answers = "|".join(z)
        paths = "(" + first_question[2] + ", " + second_question[2] + ")"
        new_data.append([new_question, answers, paths])
    
    f = open("./metaQA_create_data/metaQA_pin.txt", "w")
    for d in new_data:
        f.write("\t".join(d) + "\n")
    f.close()        
    
    return


# create_2u("./metaQA_create_data/qa_train_1hop_half_path.txt")
# create_pin("./metaQA_create_data/qa_train_2hop_half_kgc_path.txt", "./metaQA_create_data/qa_train_1hop_half_path.txt")

# train is useless, directly create test
create_2u("./metaQA_create_data/qa_train_1hop_half_path.txt")
create_pin("./metaQA_create_data/qa_train_2hop_half_kgc_path.txt", "./metaQA_create_data/qa_train_1hop_half_path.txt")


