# Path-Generator-QA 

This is a Pytorch implementation for the EMNLP 2020 (Findings) paper: 
Connecting the Dots: A Knowledgeable Path Generator for Commonsense Question Answering [[arxiv](https://arxiv.org/abs/2005.00691)]

Code folders: 

(1) `learning-generator`: conduct path sampling and then train the path generator.

(2) `commonse-qa`: use the generator to generate paths and then train the qa system on task dataset.

(3) `A-Commonsense-Path-Generator-for-Connecting-Entities.ipynb`: The notebook illustrating how to use our proposed generator to connect a pair of entities
with a commonsense relational path. 

Part of this code and instruction rely on our another project [[code](https://github.com/INK-USC/MHGRN)][[arxiv](https://arxiv.org/abs/2005.00646)].  Please cite both of our works if you use this code. Thanks!
```
@article{wang2020connecting,
  title={Connecting the Dots: A Knowledgeable Path Generator for Commonsense Question Answering},
  author={Wang, Peifeng and Peng, Nanyun and Szekely, Pedro and Ren, Xiang},
  journal={arXiv preprint arXiv:2005.00691},
  year={2020}
}

@article{feng2020scalable,
  title={Scalable Multi-Hop Relational Reasoning for Knowledge-Aware Question Answering},
  author={Feng, Yanlin and Chen, Xinyue and Lin, Bill Yuchen and Wang, Peifeng and Yan, Jun and Ren, Xiang},
  journal={arXiv preprint arXiv:2005.00646},
  year={2020}
}
```

## Dependencies

- Python >= 3.6
- PyTorch == 1.1
- transformers == 2.8.0
- dgl == 0.3 (GPU version)
- networkx == 2.3

Run the following commands to create a conda environment:

```bash
conda create -n pgqa python=3.6
source activate pgqa
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install dgl-cu100
pip install transformers==2.8.0 tqdm networkx==2.3 nltk spacy==2.1.6
python -m spacy download en
```

## For training a path generator
```bash
cd learning-generator
cd data
unzip conceptnet.zip
cd ..
python sample_path_rw.py
```

After path sampling, shuffle the resulting data './data/sample_path/sample_path.txt'
and then split them into train.txt, dev.txt and test.txt by ratio of 0.9:0.05:0.05 under './data/sample_path/'

Then you can start to train the path generator by running
```bash
# the first arg is for specifying which gpu to use
./run.sh $gpu_device
```

The checkpoint of the path generator would be stored in './checkpoints/model.ckpt'. 
Move it to '../commonsense-qa/saved_models/pretrain_generator'.
So far, we are done with training the generator.

Alternatively, you can also download our well-trained path generator from:
https://drive.google.com/file/d/1dQNxyiP4g4pdFQD6EPMQdzNow9sQevqD/view?usp=sharing.

## For training a commonsense qa system

### 1. Download Data

First, you need to download all the necessary data in order to train the model:

```bash
cd commonsense-qa
bash scripts/download.sh
```

### 2. Preprocess

To preprocess the data, run:

```bash
python preprocess.py
```

### 3. Using the path generator to connect question-answer entities 
(Modify ./config/path_generate.config to specify the dataset and gpu device)

```bash
./scripts/run_generate.sh
```

### 4. Commonsense QA system training
```bash
bash scripts/run_main.sh ./config/csqa.config
```
Training process and final evaluation results would be stored in './saved_models/'
