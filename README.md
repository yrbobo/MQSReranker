# Improving Medical Question Summarization through Re-ranking

## Folder Introduction

| Folder           | Intro                                           |
| ---------------- | ----------------------------------------------- |
| BART-FT          | Fine-tuning the BART model.                     |
| SimGate Reranker | Using SimGate Reranker as a second-stage model. |
| LLM Reranker     | Using a large language model as a reranker.     |

## Steps to Run

### 1. Fine-tuning the BART model

```shell
python finetune.py --dataset MeQSum --epoch 20 --model_save_path model/MeQSum
```

### 2. Generating candidate summaries

```shell
python candidate_generation.py --dataset MeQSum --set train --model_path model.MeQSum/your_model.pt --num_return_sequences 16
```

### 3. Train SimGate Reranker

```shell
python ReRankingMain.py --dataset MeQSum --epoch 50 --margin 0.01 --model_save_path reranking_model/MeQSum --mod train
```

### 4. Test SimGate Reranker

```shell
python ReRankingMain.py --dataset MeQSum --model_path reranking_model/MeQSum/your_model.pt --gate_threshold 0.1 
```

### 5. Test LLM as Reranker

```shell
python ChatGLMasReRanker.py --dataset MeQSum --num_cand 4
python LLaMAasReRanker.py --dataset MeQSum --numcand 4 --model LLaMA-vicuna-7B/LLaMA-vicuna-13B
```

