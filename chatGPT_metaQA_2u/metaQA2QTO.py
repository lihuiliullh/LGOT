import pickle
import ast

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
    ent_id_file = "./kbc/src/data/metaQA_half/ent_id"
    rel_id_file = "./kbc/src/data/metaQA_half/rel_id"
    ent2id, id2ent = read_dict(ent_id_file)
    rel2id, id2rel = read_dict(rel_id_file)
    
    key = (('e', ('r',)), ('e', ('r',)), ('u',))
    
    data_map = {}
    answer_map = {}
    
    data_set = set()
    
    file_name = "./chatGPT_metaQA_2u/metaQA_2u.txt"
    with open(file_name, 'r') as f:
        for line in f.readlines():
            ele = line.strip().split("\t")
            if len(ele) < 2:
                continue
            head1 = ele[0].split("[")[1].split("]")[0]
            head2 = ele[0].split("[")[2].split("]")[0]
            
            answers = ele[1].split("|")
            path = ele[2].strip()
            rel1 = path.split(",")[0][1:]
            rel2 = path.split(",")[1][0:-1].strip()
            data = ((ent2id[head1], (rel2id[rel1],)), (ent2id[head2], (rel2id[rel2],)), (-1,))
            data_set.add(data)
            answers = [ent2id[a] for a in answers]
            if data in answer_map:
                for aa in answers:
                    answer_map[data].add(aa)
            else:
                answer_map[data] = set(answers)
            a = 0
            
    #         data = (ent2id[head], (rel2id[path[0]]))
    #         data_set.add(data)
            
    #         answers = [ent2id[a] for a in answers]
    #         if data in answer_map:
    #             for aa in answers:
    #                 answer_map[data].add(aa)
    #         else:
    #             answer_map[data] = set(answers)
    
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



process()

