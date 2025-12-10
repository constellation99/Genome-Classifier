# Genome Classifier
Developed for the Knight Lab at UCSD, this project implements an end-to-end deep learning pipeline for the taxonomic classification of microbial genomes using a Transformer-based architecture. 

- Uses Biopython and NumPy to transform raw FASTA files into numerical vectors. This includes k-mer tokenization (k=6) and custom data sampling for 100,000+ sequences.

- Utilizes a custom SequenceDataset and DataLoader in PyTorch to efficiently manage and execute training jobs on GPU resources.

- Implements a rigorous validation framework using scikit-learn and Matplotlib to track performance metrics, including cross-entropy loss, ROC-AUC, and visualization of confusion matrices.
