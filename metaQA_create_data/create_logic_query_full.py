import pickle


# transform all natural language query to logic query
# all queries in the training data will be treated as training data to train KARL
# all test query data will be used in the test dataset
# all the valid data will be used in the valid data
# need to merge different queries together


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


def process_pin(ent2id, rel2id, file_name):
    
    key = (('e', ('r', 'r')), ('e', ('r', 'n')))
    
    data_map = {}
    answer_map = {}
    
    data_set = set()
    
    with open(file_name, 'r') as f:
        for line in f.readlines():
            ele = line.strip().split("\t")
            if len(ele) < 2:
                continue
            head1 = ele[0].split("[")[1].split("]")[0]
            head2 = ele[0].split("[")[2].split("]")[0]
            
            answers = ele[1].split("|")
            path = ele[2].strip()
            rel1 = path.split(",")[0][1:].strip().split("|")
            rel2 = path.split(",")[1][0:-1].strip()
            
            data = ((ent2id[head1], (rel2id[rel1[0]], rel2id[rel1[1]])), (ent2id[head2], (rel2id[rel2], -2)))
            data_set.add(data)
            
            answers = [ent2id[a] for a in answers]
            if data in answer_map:
                for aa in answers:
                    answer_map[data].add(aa)
            else:
                answer_map[data] = set(answers)
    
    data_map[key] = data_set
    return key, data_map, answer_map

def process_2u(ent2id, rel2id, file_name):
    key = (('e', ('r',)), ('e', ('r',)), ('u',))
    
    data_map = {}
    answer_map = {}
    
    data_set = set()
    
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
    data_map[key] = data_set
    return key, data_map, answer_map


def process_1p(ent2id, rel2id, file_name):
    key = ('e', ('r',))
    
    data_map = {}
    answer_map = {}
    
    data_set = set()
    
    with open(file_name, 'r') as f:
        for line in f.readlines():
            ele = line.strip().split("\t")
            if len(ele) < 2:
                continue
            head = ele[0].split("[")[1].split("]")[0]
            answers = ele[1].split("|")
            path = ele[2].split("|")
            if path[0] == "None":
                continue
            data = (ent2id[head], (rel2id[path[0]],))
            data_set.add(data)
            
            answers = [ent2id[a] for a in answers]
            if data in answer_map:
                for aa in answers:
                    answer_map[data].add(aa)
            else:
                answer_map[data] = set(answers)
    
    data_map[key] = data_set
    return key, data_map, answer_map


def process_2p(ent2id, rel2id, file_name):
    key = ('e', ('r', 'r'))
    
    data_map = {}
    answer_map = {}
    
    data_set = set()
    
    with open(file_name, 'r') as f:
        for line in f.readlines():
            ele = line.strip().split("\t")
            if len(ele) < 2:
                continue
            head = ele[0].split("[")[1].split("]")[0]
            answers = ele[1].split("|")
            path = ele[2].split("|")
            
            data = (ent2id[head], (rel2id[path[0]], rel2id[path[1]]))
            data_set.add(data)
            
            answers = [ent2id[a] for a in answers]
            if data in answer_map:
                for aa in answers:
                    answer_map[data].add(aa)
            else:
                answer_map[data] = set(answers)
    
    data_map[key] = data_set
    return key, data_map, answer_map


def process_3p(ent2id, rel2id, file_name):
    key = ('e', ('r', 'r', 'r'))
    
    data_map = {}
    answer_map = {}
    
    data_set = set()
    
    with open(file_name, 'r') as f:
        for line in f.readlines():
            ele = line.strip().split("\t")
            if len(ele) < 2:
                continue
            head = ele[0].split("[")[1].split("]")[0]
            answers = ele[1].split("|")
            path = ele[2].split("|")
            if head == "None":
                continue
            data = (ent2id[head], (rel2id[path[0]], rel2id[path[1]], rel2id[path[2]]))
            data_set.add(data)
            
            answers = [ent2id[a] for a in answers]
            if data in answer_map:
                for aa in answers:
                    answer_map[data].add(aa)
            else:
                answer_map[data] = set(answers)
                
    data_map[key] = data_set
    return key, data_map, answer_map



def add_second_map_2_first_map(map1, map2):
    for k, v in map2.items():
        map1[k] = v
    return map1

# no training for pin, 2u (inductive setting, also not used them to train QTO or query2box, how could you used pin to test query2box?)
def create_train():
    train_1hop = "./chatGPT_metaQA_1hop/qa_train_1hop_path.txt"
    train_2hop = "./chatGPT_metaQA_2hop/qa_train_2hop_path.txt"
    train_3hop = "./chatGPT_metaQA_3hop/qa_train_3hop_path.txt"
    
    # make dataset according to the natural language information
    ent_id_file = "./kbc/src/data/metaQA_full/ent_id"
    rel_id_file = "./kbc/src/data/metaQA_full/rel_id"
    ent2id, id2ent = read_dict(ent_id_file)
    rel2id, id2rel = read_dict(rel_id_file)
    
    key1, data_map1, answer_map1 = process_1p(ent2id, rel2id, train_1hop)
    key2, data_map2, answer_map2 = process_2p(ent2id, rel2id, train_2hop)
    key3, data_map3, answer_map3 = process_3p(ent2id, rel2id, train_3hop)
    data_map1 = add_second_map_2_first_map(data_map1, data_map2)
    data_map1 = add_second_map_2_first_map(data_map1, data_map3)
    answer_map1 = add_second_map_2_first_map(answer_map1, answer_map2)
    answer_map1 = add_second_map_2_first_map(answer_map1, answer_map3)
    
    data_map = data_map1
    answer_map = answer_map1
    answer_map_hard = answer_map1
    # write file to pickle
    with open('train-queries.pkl', 'wb') as handle:
        pickle.dump(data_map, handle)
    with open('train-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map_hard, handle)
    # with open('train-hard-answers.pkl', 'wb') as handle:
    #     pickle.dump(answer_map, handle)
    
    with open('id2ent.pkl', 'wb') as handle:
        pickle.dump(id2ent, handle)
    with open('ent2id.pkl', 'wb') as handle:
        pickle.dump(ent2id, handle)
    with open('id2rel.pkl', 'wb') as handle:
        pickle.dump(id2rel, handle)
    

def create_test():
    test_pin_query_file = "./chatGPT_metaQA_pin/metaQA_pin.txt"
    test_2u_query_file = "./chatGPT_metaQA_2u/metaQA_2u.txt"
    test_1hop_query_file = "./chatGPT_metaQA_1hop/qa_test_1hop_path.txt"
    test_2hop_query_file = "./chatGPT_metaQA_2hop/qa_test_2hop_path.txt"
    test_3hop_query_file = "./chatGPT_metaQA_3hop/qa_test_3hop_path.txt_rz"
    
    # make dataset according to the natural language information
    ent_id_file = "./kbc/src/data/metaQA_full/ent_id"
    rel_id_file = "./kbc/src/data/metaQA_full/rel_id"
    ent2id, id2ent = read_dict(ent_id_file)
    rel2id, id2rel = read_dict(rel_id_file)
    
    key1, data_map1, answer_map1 = process_1p(ent2id, rel2id, test_1hop_query_file)
    key2, data_map2, answer_map2 = process_2p(ent2id, rel2id, test_2hop_query_file)
    key3, data_map3, answer_map3 = process_3p(ent2id, rel2id, test_3hop_query_file)
    key4, data_map4, answer_map4 = process_pin(ent2id, rel2id, test_pin_query_file)
    key5, data_map5, answer_map5 = process_2u(ent2id, rel2id, test_2u_query_file)
    
    data_map1 = add_second_map_2_first_map(data_map1, data_map2)
    data_map1 = add_second_map_2_first_map(data_map1, data_map3)
    data_map1 = add_second_map_2_first_map(data_map1, data_map4)
    data_map1 = add_second_map_2_first_map(data_map1, data_map5)
    answer_map1 = add_second_map_2_first_map(answer_map1, answer_map2)
    answer_map1 = add_second_map_2_first_map(answer_map1, answer_map3)
    answer_map1 = add_second_map_2_first_map(answer_map1, answer_map4)
    answer_map1 = add_second_map_2_first_map(answer_map1, answer_map5)
    
    data_map = data_map1
    answer_map = answer_map1
    answer_map_hard = answer_map1
    # write to pickle
    with open('test-queries.pkl', 'wb') as handle:
        pickle.dump(data_map, handle)
    with open('test-easy-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map_hard, handle)
    with open('test-hard-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map, handle)
    
     # write to pickle
    with open('valid-queries.pkl', 'wb') as handle:
        pickle.dump(data_map, handle)
    with open('valid-easy-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map_hard, handle)
    with open('valid-hard-answers.pkl', 'wb') as handle:
        pickle.dump(answer_map, handle)
    
    with open('id2ent.pkl', 'wb') as handle:
        pickle.dump(id2ent, handle)
    with open('ent2id.pkl', 'wb') as handle:
        pickle.dump(ent2id, handle)
    with open('id2rel.pkl', 'wb') as handle:
        pickle.dump(id2rel, handle)
    with open('rel2id.pkl', 'wb') as handle:
        pickle.dump(rel2id, handle)



def transform_file(in_file_name, out_file_name, ent2id, rel2id):
    with open(out_file_name, 'w') as o:
        with open(in_file_name, 'r') as f:
            for line in f.readlines():
                ele = line.strip().split("\t")
                if len(ele) < 2:
                    continue
                head_id = ent2id[ele[0]]
                rel_id = rel2id[ele[1]]
                tail_id = ent2id[ele[2]]
                o.write(str(head_id) + "\t" + str(rel_id) + "\t" + str(tail_id) + "\n")
        f.close()
    o.close()
            
    
def create_text():
    ent_id_file = "./kbc/src/data/metaQA_full/ent_id"
    rel_id_file = "./kbc/src/data/metaQA_full/rel_id"
    ent2id, id2ent = read_dict(ent_id_file)
    rel2id, id2rel = read_dict(rel_id_file)
    
    train = "./kbc/src/src_data/metaQA_full/train"
    test = "./kbc/src/src_data/metaQA_full/test"
    valid = "./kbc/src/src_data/metaQA_full/valid"
    
    transform_file(train, train + ".txt", ent2id, rel2id)
    transform_file(test, test + ".txt", ent2id, rel2id)
    transform_file(valid, valid + ".txt", ent2id, rel2id)


create_text()
create_test()
create_train()
