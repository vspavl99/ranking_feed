# ranking-feed
Repository for experiments of recommender systems for ranking feed 2022/2023

Experiments in [wandb](https://wandb.ai/theeneus/recbole?workspace=user-theeneus)

## Baseline models

### mind_small
| Model\Metric | MAP@10 | NDCG@10 | Recall@10 | Precision@10 |  
|--------------|--------|---------|-----------|--------------|
| Pop          | 0.078  | 0.1651  | 0.0976    | 0.1598       |
| BPR          | 0.0777 | 0.1381  | 0.074     | 0.1254       |
| DMF          | 0.3029 | 0.4199  | 0.1357    | 0.3682       |
| MultiVAE     | 0.3864 | 0.5375  | 0.2421    | 0.4902       |

### mind_large
| Model\Metric | MAP@10 | NDCG@10  | Recall@10 | Precision@10 |  
|--------------|--------|----------|-----------|--------------|
| Pop          | 0.1185 | 0.2252   | 0.11      | 0.2174       |
| BPR          | 0.1679 | 0.2612   | 0.103     | 0.234        |
| DMF          |        |          |           |              |
| MultiVAE     |        |          |           |              |


## External data

### mind_small
| Model\Metric  | MAP@10 | NDCG@10 | Recall@10 | Precision@10 |  
|---------------|--------|---------|-----------|--------------|
| MultiVAE      | 0.1102 | 0.2134  | 0.1104    | 0.2078       |
| MultiVAE exp1 | 0.0899 | 0.1744  | 0.1021    | 0.1664       |
| DSSM          | 0.1843 | 0.3282  | 0.1663    | 0.2996       |

* Exp1: Использование эмбедингов для инициализации последнего линейного слоя в Decoder