import pickle
import traceback

def read_dict(entity_dict_file):
    e2id = {}
    id2e = {}
    f = open(entity_dict_file, 'r')
    for line in f:
        line = line.strip().split('\t')
        ent_name = line[0]
        ent_id = line[1]
        ent_id = int(ent_id)
        e2id[ent_name] = ent_id
        id2e[ent_id] = ent_name
    f.close()

    return e2id, id2e

def process():
    ent_id_file = "./kbc/src/data/GraphQuestionsFull/ent_id"
    rel_id_file = "./kbc/src/data/GraphQuestionsFull/rel_id"
    
    ent2id, id2ent = read_dict(ent_id_file)
    rel2id, id2rel = read_dict(rel_id_file)
    
    key = ('e', ('r', 'r'))
    
    data_map = {}
    answer_map = {}
    
    data_set = set()
    
    file_name = "chatGPT_GraphQuestions/GraphQuestions_good_2hop_test.pkl"
    with open(file_name, 'rb') as f:
        datas_test = pickle.load(f)
        
    file_name_train = "chatGPT_GraphQuestions/GraphQuestions_good_2hop_train.pkl" # merge train and test together, because the data is too few. 
    with open(file_name_train, 'rb') as f:
        datas = pickle.load(f)
        datas.extend(datas_test)
        
        #datas = datas_test
        for line in datas:
            question_text = line[0]
            entity1 = line[1][0]
            entity2 = line[1][1]
            ans = line[1][2]
            rel1 = line[1][3]
            rel2 = line[1][4]
            
            try:
                data = (ent2id[entity1[0]], (rel2id[rel1], rel2id[rel2]))
            except:
                print(line)
                traceback.print_exc()
                continue
            data_set.add(data)
            
            answers = [ent2id[a['answer_argument']] for a in ans]
            if data in answer_map:
                for aa in answers:
                    answer_map[data].add(aa)
            else:
                answer_map[data] = set(answers)
    
    data_map[key] = data_set
    
    # write to pickle
    with open('test-queries.pkl', 'wb') as handle:
        pickle.dump(data_map, handle)
    with open('test-easy-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map, handle)
    with open('test-hard-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map, handle)
    
     # write to pickle
    with open('valid-queries.pkl', 'wb') as handle:
        pickle.dump(data_map, handle)
    with open('valid-easy-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map, handle)
    with open('valid-hard-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map, handle)
    
    
    with open('id2ent.pkl', 'wb') as handle:
        pickle.dump(id2ent, handle)
    with open('ent2id.pkl', 'wb') as handle:
        pickle.dump(ent2id, handle)
    with open('id2rel.pkl', 'wb') as handle:
        pickle.dump(id2rel, handle)


def txt_triplet2id_triplet():
    ent_id_file = "./kbc/src/data/freebase_GraphQuestions/ent_id"
    rel_id_file = "./kbc/src/data/freebase_GraphQuestions/rel_id"
    
    ent_id_file = "./kbc/src/data/GraphQuestionsFull/ent_id"
    rel_id_file = "./kbc/src/data/GraphQuestionsFull/rel_id"
    
    ent2id, id2ent = read_dict(ent_id_file)
    rel2id, id2rel = read_dict(rel_id_file)
    
    data = ['train', 'valid', 'test']
    file = "./kbc/src/src_data/freebase_GraphQuestions/"
    file = "./kbc/src/src_data/GraphQuestionsFull/"
    
    for d in data:
        id_data = []
        with open(file + d, 'r') as f:
            for line in f.readlines():
                ele = line.strip().split("\t")
                if len(ele) < 2:
                    continue
                id_data.append([ent2id[ele[0]], rel2id[ele[1]], ent2id[ele[2].strip()]])
        f.close()
        
        # write to file
        with open(d + ".txt", "w") as f:
            for e in id_data:
                f.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")
        f.close()


process()
#txt_triplet2id_triplet()

