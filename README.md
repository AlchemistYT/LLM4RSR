# LLM4RSR: Large Language Models as Data Correctors for Robust Sequential Recommendation

## Requirements
- `numpy==1.24.2`
- `scipy==1.10.1`
- `torch==2.0.0`
- `python=3.8.10`
- `CUDA==11.4`
- `seaborn==0.12.2`
- `apex==0.9.10dev`
- `matplotlib==3.7.1`

## Usage
1. Install required packages.
2. Run <code>pip install -e .</code> to install dependencies required by Llama2.
3. Run <code>download.sh</code> to install Llama2-7B (detailed instructions can be found here: https://github.com/Meta-Llama/llama).
4. Run <code>nohup torchrun --nproc_per_node 1 tgd.py</code> to optimize the prompts via TGD on various datasets. 
5. Run <code>nohup torchrun --nproc_per_node 1 ml1m_inference.py</code> to run the optimized prompts on ML-1M dataset. Similar scripts are available for
   - CD: <code>nohup torchrun --nproc_per_node 1 cd_inference.py</code>
   - Game: <code>nohup torchrun --nproc_per_node 1 game_inference.py</code>
   - Kindle: <code>nohup torchrun --nproc_per_node 1 kindle_inference.py</code>


## Datasets
- All the datasets used in our paper are organized in [dataset/](dataset/), where each data dir contains three files:
  - i_idx2str.dat stores the attributes of each item
  - i_idx2summary stores the summary of each item
  - synthetic_instances_simple.dat stores the constructed TGD dataset
- ML-1M is from https://grouplens.org/datasets/movielens/1m/,
- CD, Game, and Kindle are from http://jmcauley.ucsd.edu/data/amazon/.



## Codes for backbones
The optimized prompts can be used to correct the datasets of SRs:
- GRU4Rec: https://github.com/hidasib/GRU4Rec
- Caser: https://github.com/graytowne/caser_pytorch
- SASRec: https://github.com/kang205/SASRec
- LRURec: https://github.com/yueqirex/LRURec
- RecFormer: https://github.com/AaronHeee/RecFormer
- BERD+: https://bitbucket.org/SunYatong/berdplus-tois-2023/src/master/
- FMLP-Rec: https://github.com/Woeee/FMLP-Rec
- STEAM: https://github.com/tempsdu/steam
- BirDRec: https://github.com/AlchemistYT/BirDRec
- SSDRec: https://github.com/zc-97/SSDRec
