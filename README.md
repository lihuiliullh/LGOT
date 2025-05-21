All the dataset can be found from https://drive.google.com/drive/folders/1H1XHJKRv1rI1FGC4W3nvJIpPvmtfEpcN?usp=share_link

Because the github does not support big files, so we store all the big files and model parameters on google driver

---------------

To run LGOT

1. store your own kg in kbc/src/src_data (this is where KG stores) (KG will be used by embedding based KG retrieval)

Because Freebase stores entity as entity_id (e.g., m.0490fcx), we replace all entity_id with its true name (e.g., West virginia)


2. cd kbc/src, run python preprocess_datasets.py 

This step will process dataset, generate id for all entities and relations. The output is kbc/src/data


3. Pretrain KGC model (calculate the KG embeddings)

under kbc/src, run the following command

python main.py --dataset FB15K --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 100 --lmbda 0.01 --w_rel 0.1 --max_epochs 100

python main.py --dataset FB15K-237 --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 100

python main.py --dataset NELL995 --score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 0 --max_epochs 100

note that after training, the embedding file will stored in kbc/{dataset}


4. Run lgot, query data preparation (metaQA does not have 2u, pin and other query types, so we need to generate the query by ourself) (the generated data can be found in the repository)

before run lgot, all the query needs to be transformed from natural language query to logical query graph, so that fuzzy logic can be used.

The code is stored in ./metaQA_create_data

For 1-hop, 2-hop, 3-hop, we directly used a subset of the original metaQA data (use do not use all of them, because the dataset is very large. ChatGPT will takes a lot of time to find the answers)

we create 2u, pin, the data is stored in ./metaQA_create_data (notice that all baselines, e.g., query2box, QTO only trained on 1-hop, 2-hop, 3-hop queries, so we do not need to create 2u, pin training data). 2u, pin queries are used to test the generalization ablity of the models.

we run create logical queries and store all the pickle file in ./data/xxx


5. after create query, run lgot

under ./LGOT

It stores all the ablation study code.


main_metaQA.py is for 1-hop, 2-hop, 3-hop
main_metaQA_pin.py is for pin
main_metaQA_2u.py is for 2u

folder chatGPT_metaQA_xxxx stores the natural language query for different queries





6. For CWQ dataset. the train, valid, test queries are stored in ./chatGPT_CWQ/CWQ_good_3hop_xxxx.pickle

(GraphQuestions uses different code framework but the same idea as metaQA and CWQ)
7. For GraphQuestions. the dataset and code are stored in ./chatGPT_GraphQuestions and ./chatGPT_GraphQuestions2i



In order to make the dataset easy to use, I copy all the necessary queries and kg to folder dataset. 


If you think the paper and code is useful. please consider cite our paper 


      @misc{liu2024logicquerythoughtsguiding,

      title={Logic Query of Thoughts: Guiding Large Language Models to Answer Complex Logic Queries with Knowledge Graphs}, 
      
      author={Lihui Liu and Zihao Wang and Ruizhong Qiu and Yikun Ban and Eunice Chan and Yangqiu Song and Jingrui He and Hanghang Tong},
      
      year={2024},
      
      eprint={2404.04264},
      
      archivePrefix={arXiv},
      
      primaryClass={cs.IR},
      
      url={https://arxiv.org/abs/2404.04264}, 
      
      }
'''
M4
