# Pocket hopping: A Drug Repositioning Method Based On Binding Site Comparison of Co-ligand Protein Structures Via Graph Attention Network
Pocket hopping is a protein "pocket hopping" approach for drug repurposing, targeting proteins with distinct structures yet shared (or highly similar) ligands. Centered on ligand-binding pockets, a graph attention neural network is employed to learn associations among such proteins. This model consistently outperforms conventional alternatives in external evaluations, successfully identifying structurally and evolutionarily unrelated new targets for existing drugs while pinpointing key amino acids underlying the "hopping" process. It is anticipated to deliver valuable support for structure-based drug design and the advancement of drug repurposing research.

<img width="1604" height="750" alt="ab01a2e4-f91e-4ca1-bc2f-9651e200f7b4" src="https://github.com/user-attachments/assets/39dc76a5-5a4a-4d9c-9eee-275c04e5f656" />

# The automated Pocket Hopping screening pipeline
Consisting of the construction of the P-S-L (Protein–Site–Ligand) database and a target-adaptable automated workflow. First, the database is built by integrating high-quality curated binding sites from PDB complexes with the expanded activity-annotated chemical space of DrugSpaceX, encompassing not only known bioactive ligands but also a large number of computationally generated novel analogs, which breaks the limitations of traditional ligand redirection. For any given target protein, the pipeline automatically identifies its binding site, encodes it into a graph-based pocket representation, uses the Pocket Hopping model to predict non-homologous pockets with similar ligand-recognition patterns, and further retrieves relevant active ligands and their derived analogs from the P-S-L database to form a candidate compound library for downstream experimental evaluation—providing an efficient and innovative cross-target screening strategy for drug discovery.
<img width="940" height="646" alt="image" src="https://github.com/user-attachments/assets/382fdcf9-fc57-41fd-91f4-ae34132bb524" />


# Description

**Things Pocket hopping can do**
- Protein similarity comparison
- Drug repositioning

# Setup Environment
### 1. Clone the current repo
```bash
git clone https://github.com/niubuying/PocketHop.git
```
### 2.Installation
```bash
conda create -n PocketHop python=3.7
conda activate PocketHop
conda install pytorch torchvision torchaudio -c conda-forge
pip install scikit-learn
…
```

# Running Pocket hopping
### 1.Data acquisition via pdb database:
```bash
cd data_preparation
python fetch_pdbs.py
```

Generate molecular graphs from the collected data
```bash
cd data_preparation
python generate_graphs.py
```

### 2.Data preprocessing:
```bash
python site_featurizer.py
```
### 3.Train:
```bash
python training_graphbsm.py
```
### 4.Predict:
```bash
python predict_graphbsm.py
```
## Related data and model are saved below:
https://zenodo.org/records/17572018
