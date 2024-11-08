import json
import pickle

def check():
    file_name = "./chatGPT_GraphQuestions2i/evidence_2i_GraphQuestions.txt"
    
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 80):
            ans_set = set()
            if i + 80 > len(lines):
                break
            for j in range(i, i+80, 4):
                first_line = lines[j]
                second_line = lines[j+1]
                third_line = lines[j+2]
                fourth_line = lines[j+3]
                
                third_line = third_line.strip()
                third_line = third_line.split(" ")
                ansss = third_line[0]
                
                result_map = json.loads(fourth_line)
                
                ansss = result_map[ansss]
                ans_set.add(ansss)
    a = 0


def process():
    file_name = "./chatGPT_GraphQuestions2i/evidence_2i_GraphQuestions.txt"
    res_final = {}
    with open(file_name, 'r') as f:
        lines = f.readlines()
        
        for i in range(0, len(lines), 80):
            ans_set = set()
            if i + 80 > len(lines):
                break
            for j in range(i, i+80, 4):
                first_line = lines[j]
                second_line = lines[j+1]
                third_line = lines[j+2]
                fourth_line = lines[j+3]
                
                second_line = second_line.strip()
                second_line = " ".join(second_line.split())
                third_line = third_line.strip()
                third_line = " ".join(third_line.split())
                
                second_line = second_line.split(" ")
                third_line = third_line.split(" ")
                
                r1_out = second_line[2].split("-")[1]
                r2_out = third_line[1].split("-")[1]
                
                entity1 = second_line[3]
                entity2 = third_line[2]
                

                ansss = third_line[0]
                
                result_map = json.loads(fourth_line)
                
                entity1 = result_map[entity1]
                entity2 = result_map[entity2]
                r1_out = result_map[r1_out]
                r2_out = result_map[r2_out]
                
                key = (entity1, entity2, r1_out, r2_out)
                
                truthfulness = first_line.split(" ")[0]
                if truthfulness != "False":
                    continue
                
                ansss = result_map[ansss]
                ans_set.add(ansss)
                 
            res_final[key] = ans_set
    
    a = 0
    # write to file
    with open('./chatGPT_GraphQuestions2i/false_evidence_2i_GraphQuestions.pkl', 'wb') as handle:
        pickle.dump(res_final, handle)


check()
process()

