# MGMA-PPIS

## 1 Description
	MGMA-PPIS is a novel graph neural network-based method to predict PPI sites by adopting Multi-view Graph embedding and Multi-scale Attention fusion.

## 2 environment requirements
	(1) python 3.9
	(2) torch-2.3.0+cu121
	(3) dgl 2.4.0.th23.cu121
	(4) pandas
	(5) sklearn
 
## 3 Datasets
	The files in "./Dataset" include the datasets used in this experiment are the original datasets from [AGAT-PPIS]( https://github.com/AILBC/AGAT-PPIS ).
	All the processed pdb files of the protein chains used in this experiment are put in the directory "./Dataset/pdb/".
## 4 Features
	The extracted features are in the directory "./Feature". The specific meanings are listed as follows.
	(1) PSSM: the PSSM matrix of the protein chains used in this experiment.
	(2) HMM: the HMM matrix of the protein chains used in this experiment.
	(3) DSSP: the DSSP matrix of the protein chains used in this experiment.
	(4) AF: the atom features of the residues for each protein used in the experiment.
	(5) PPE: the resiude pseudo positions of the protein chains in those datasets.
       
## 5 Usage
	The construction of the model is in the " MGMA_model.py".<br>
 	You can run "train.py" to train the deep model from stratch and use the "test.py" to test the test datasets with the trained model.

  
## License
This dataset is released under the [CC0 1.0 Universal (CC0 1.0)](LICENSE) Public Domain Dedication license.

