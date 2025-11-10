# Pocket hopping: A Drug Repositioning Method Based On Binding Site Comparison of Co-ligand Protein Structures Via Graph Attention Network
Pocket hopping is a protein "pocket hopping" approach for drug repurposing, targeting proteins with distinct structures yet shared (or highly similar) ligands. Centered on ligand-binding pockets, a graph attention neural network is employed to learn associations among such proteins. This model consistently outperforms conventional alternatives in external evaluations, successfully identifying structurally and evolutionarily unrelated new targets for existing drugs while pinpointing key amino acids underlying the "hopping" process. It is anticipated to deliver valuable support for structure-based drug design and the advancement of drug repurposing research.
<img width="1604" height="750" alt="ab01a2e4-f91e-4ca1-bc2f-9651e200f7b4" src="https://github.com/user-attachments/assets/39dc76a5-5a4a-4d9c-9eee-275c04e5f656" />

# Description

**Things Pocket hopping can do**
- Protein similarity comparison
- Drug repositioning

# Setup Environment
## 1. Clone the current repo
```bash
git clone https://github.com/niubuying/PocketHop.git
```
## 2.Installation
```bash
conda create -n PocketHop python=3.7
conda activate PocketHop
conda install pytorch torchvision torchaudio -c conda-forge
pip install scikit-learn
…
```

# Running Pocket hopping
## 1.Data preprocessing:
```bash
python create_data.py
```
## 2.Train:
```bash
python training_graphbsm.py
```
## 3.Predict:
```bash
python predict_graphbsm.py
```
## 4.Related data and model are saved below:
https://zenodo.org/records/17572018
