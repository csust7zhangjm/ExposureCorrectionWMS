# Partition-based Image Exposure Correction
Jianming Zhang, Jia Jiang, Mingshuang Wu, Zhijian Feng, Xiangnan Shi

[paper](paper%20link)

## Setup
Install required packages 
``
pip install -r requirements.txt
``
# How to Run
## Test
1. Prepare data: set the '--test_LL_folder' in evaluation.py to your own path. 
2. Prepare pretrained model: Put pretrained model under:  ``./checkpoints/``
3. Run:  python evaluation.py
## Train
1. Prepare data: set the '--train_LL_folder' and '--img_val_path' in train.py to your train dataset and validation dadaset.
2. Run: python train.py
# Dataset & Pretained Model
1. The LOL-v1 and LOL-v2 datasets analyzed during the current study are available in [LOL](https://daooshee.github.io/BMVC2018website/)
2. The MIT-Adobe FiveK dataset analyzed during the current study is available in [5k](https://data.csail.mit.edu/graphics/fivek/)
3. The MSEC dataset is available in [msec](https://github.com/mahmoudnafifi/Exposure_Correction).
4. The LCDP dataset is available in [lcdp](https://hywang99.github.io/2022/07/09/lcdpnet/)
5. The VV, LIME, NPE and DICM datasets analyzed during the current study are available in [non-reference datasets](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T).
We provie pretained models:
``checkpoints/best_Epoch_LOL.pth``
``checkpoints/best_Epoch_LCDP.pth``
``checkpoints/me/best_EpochS.pth``
# Cite This Paper
If you find our work or code helpful, or your research benefits from this repo, please cite our paper.
@article{ZHANG2024104342,
title = {Illumination-guided dual-branch fusion network for partition-based image exposure correction},
journal = {Journal of Visual Communication and Image Representation},
pages = {104342},
year = {2024},
issn = {1047-3203},
doi = {https://doi.org/10.1016/j.jvcir.2024.104342},
url = {https://www.sciencedirect.com/science/article/pii/S1047320324002980},
author = {Jianming Zhang and Jia Jiang and Mingshuang Wu and Zhijian Feng and Xiangnan Shi}}