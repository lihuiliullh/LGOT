import pickle


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
    ent_id_file = "./kbc/src/data/freebase_CWQ/ent_id"
    rel_id_file = "./kbc/src/data/freebase_CWQ/rel_id"
    ent_id_file = "./kbc/src/data/CWQFull/ent_id"
    rel_id_file = "./kbc/src/data/CWQFull/rel_id"
    
    ent2id, id2ent = read_dict(ent_id_file)
    rel2id, id2rel = read_dict(rel_id_file)
    
    key = ('e', ('r', 'r', 'r'))
    
    data_map = {}
    answer_map = {}
    
    data_set = set()
    
    file_name = "chatGPT_CWQ/CWQ_good_3hop_train.pickle"
    with open(file_name, 'rb') as f:
        datas = pickle.load(f)
        for line in datas:
            head = line[1]
            answers = line[3]
            path = line[2]
            
            try:
                data = (ent2id[head], (rel2id[path[0]], rel2id[path[1]], rel2id[path[2]]))
            except:
                print(line)
                continue
            
            
            
            new_answerss = []
            for aaaaa in answers:
                if aaaaa in ent2id:
                    new_answerss.append(ent2id[aaaaa])
            answers = new_answerss
            
            if len(answers) == 0:
                continue
            
            data_set.add(data)
            
            if data in answer_map:
                for aa in answers:
                    answer_map[data].add(aa)
            else:
                answer_map[data] = set(answers)
    
    data_map[key] = data_set
    
    # write to pickle
    with open('train-queries.pkl', 'wb') as handle:
        pickle.dump(data_map, handle)
    with open('train-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map, handle)



process()

