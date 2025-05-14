#python3.9 -m experiments.test_grand --dataset PROTEINS --folds 10 --epochs 20 --K 3 --dropnode 0.5 --dropout 0.5
#python3.9 -m experiments.test_grandpp --dataset PROTEINS --folds 3 --epochs 20 --K 3 --dropnode 0.5 --dropout 0.5 --views 4 --lambda_cons 1.0
#python3.9 -m experiments.test_gat --folds 10 --epochs 10  --layers 5 --dataset ENZYMES
#python3.9 -m experiments.test_mpnn --folds 3 --epochs 10 --layers 3 --dataset MUTAG
#python3.9 -m experiments.test_gcn --folds 3 --epochs 20 --layers 5 --dataset PROTEINS
#python3.9 -m experiments.test_graphsage --folds 10 --epochs 20 --layers 3 --dataset PROTEINS
python3.9 -m experiments.test_gin --folds 10 --epochs 10 --layers 3 --dataset PROTEINS