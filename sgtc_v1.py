# IMPORTS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import random
import math
from itertools import product
import gzip
import io
import os
import glob
from Bio import SeqIO, Entrez
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from collections import Counter

# GENERATING DATA
# Generates a list of random subsequences of a given length from a FASTA file
def generate_random_sample(filepath, length=150, num_subsequences=10000):
    # Combine all record sequences in the file
    full_seq = ""
    
    # Parse fasta files
    for record in SeqIO.parse(filepath, "fasta"):
        full_seq += str(record.seq)
    #with gzip.open(filepath, 'rt') as handle:
        #for record in SeqIO.parse(handle, "fasta"):
            #full_seq += str(record.seq)

    # Check that the sequence is long enough
    if len(full_seq) < length:
        raise ValueError("The sequence is too short.")

    samples = []
    organism_name = filepath.split('/')[-1].replace('.fasta', '')

    # Generate random subsequences
    for _ in range(num_subsequences):
        # Choose a random start position and append the corresponding subsequence to samples
        start = random.randint(0, len(full_seq) - length)
        subseq = full_seq[start:start+length]
        samples.append((subseq, organism_name))

    return samples


# Gets all of the .fna files in a specified directory
def get_fna_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.fna')]
    

directory = '/home/ajcagle/genome_files/10_phyla'
files = get_fna_files(directory)

sequences = []
labels = []

for file in files:
  for subseq, label in generate_random_sample(file, num_subsequences=10000):
    sequences.append(subseq)
    labels.append(label)

sequences, labels = shuffle(sequences, labels, random_state=42)

unique_labels = set(labels)
print("Number of Labels:", len(unique_labels), '\n')
print("Unique Labels:", unique_labels, '\n')

print("Number of Sequences:", len(sequences), '\n')


# PREPROCESSING
# Makes k-mers
# Breaks up sequences into subsequences with length k
# using sliding window and list comprehension
def make_kmers(sequences, k):    
    return [[seq[j:j+k] for j in range(len(seq) - k + 1)] for seq in sequences]

# Generates dictionary with every possible combination of k-mers
# The dictionary will have size 4^k
# e.g. for a 3-mer, this would be 1 through 4^3=64
def generate_kmer_dict(k):
    nucleotides = ['A', 'C', 'G', 'T']
    kmer_dict = {''.join(kmer): i+1 for i, kmer in enumerate(product(nucleotides, repeat=k))}
    return kmer_dict

# Encodes strings as integers
# Goes through list of kmers from each sequence, converts k-mers to corresponding numerical values from kmer_dict
def encode_kmers(kmers, kmer_dict):
    return [[kmer_dict[kmer] for kmer in kmer_subset] for kmer_subset in kmers]

# Filters data for valid sequences
# Removes any sequences that contain a base other than A, C, T, or G
def valid_sequences(sequences, labels):
    if len(sequences) != len(labels):
        raise ValueError("Sequences and labels must be the same length.")

    valid_bases = {'A', 'T', 'C', 'G'}
    
    filtered_data = [(seq, lab) for seq, lab in zip(sequences, labels) if all(base in valid_bases for base in seq)]
    filtered_sequences, filtered_labels = zip(*filtered_data) if filtered_data else ([], [])
    
    return filtered_sequences, filtered_labels

# Combines preprocessing steps into one function call
def preprocess_sequence(sequence, k=6):
    # Makes k-mers
    kmers = make_kmers([sequence], k)[0]

    # Generates k-mer dictionary
    kmer_dict = generate_kmer_dict(k)

    # Encodes k-mers
    encoded_kmers = encode_kmers([kmers], kmer_dict)[0]

    return encoded_kmers


# MODEL
# Creating transformer model to classify genomic sequences
class GenomeClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, num_classes, max_seq_len=200, device='cpu'):
        super(GenomeClassifier, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1),
            num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model) 
        out = self.transformer(src)
        out = out.mean(dim=1)
        out = self.fc(out)
        return out


# CUSTOM DATASET
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, label_encoder, k=6):
        # Filter out invalid sequences
        sequences, labels = valid_sequences(sequences, labels)

        self.sequences = sequences
        self.labels = labels
        self.k = k
        self.label_encoder = label_encoder
        self.labels_encoded = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Preprocess sequences
        sequence_encoded = preprocess_sequence(sequence, self.k)
        label = self.labels_encoded[idx]

        return torch.tensor(sequence_encoded), torch.tensor(label)
    

# Get Taxonomy
Entrez.email = "ajcagle@ucsd.edu"

def get_gcf_id(filename):
    gcf_id = filename.split('_')[0] + '_' + filename.split('_')[1]
    print(gcf_id)
    return gcf_id

def get_taxonomy(gcf_id):
    with Entrez.esearch(db="assembly", term=gcf_id) as handle:
        record = Entrez.read(handle)
        assembly_id = record["IdList"][0]

    with Entrez.esummary(db="assembly", id=assembly_id) as handle:
        summary = Entrez.read(handle)

    return summary['DocumentSummarySet']['DocumentSummary'][0]['SpeciesName']


taxa_mapping = {}

print('\n')
for file in files:
    gcf_id = get_gcf_id(os.path.basename(file))
    taxonomy = get_taxonomy(gcf_id)
    print("File Name:", os.path.basename(file))
    print("GCF ID:", gcf_id)
    print("Species Name:", taxonomy)
    print('\n')
    taxa_mapping[os.path.basename(file)] = taxonomy
    time.sleep(0.5)

print(taxa_mapping, '\n')


# VISUALIZATION
# Heatmap of model performance
def plot_confusion_matrix(true_labels, pred_labels, species_names, is_training=True):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(40,22))
    sns.heatmap(cm, annot=False, fmt="d", cmap=plt.cm.Greens, square=True, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=46)
    ax.set_ylabel("True Label", fontsize=46)

    if is_training:
        ax.set_title("Validation Confusion Matrix", fontsize=48)
    else:
        ax.set_title("Test Confusion Matrix", fontsize=48)
    
    plt.xticks(rotation=45, ha="right", fontsize=40)
    plt.yticks(fontsize=40, rotation=45)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=40)
    
    plt.subplots_adjust(bottom=0.3, left=0.4)
    
    for idx, sp in enumerate(species_names):
        plt.scatter([], [], color='gray', label=f"{idx}: {sp}", s=50)
    
    plt.legend(title="Labels", fontsize=40, title_fontsize=42, loc='center left', bbox_to_anchor=(1.2, 0.5))
    plt.tight_layout()

    ax.tick_params(axis='both', which='major', width=3, length=10)

    if is_training:
        plt.savefig("conf_matx_val.png", format="png", dpi=300)
    else:
        plt.savefig("conf_matx_test.png", format="png", dpi=300)

    plt.show()

# Plot of loss vs accuracy for training and validation data
def plot_loss_accuracy_curve(train_losses, train_accuracies, val_losses, val_accuracies, num_epochs):
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='orange', linewidth=3.0)
    plt.plot(epochs, val_losses, label='Validation Loss', color='blue', linewidth=3.0)
    plt.legend(fontsize=32, loc='upper right')
    plt.title('Losses Over Epochs', fontsize=38)
    plt.xlabel("Epochs", fontsize=36)
    plt.ylabel("Loss", fontsize=36)
    plt.xticks(ticks=epochs, fontsize=32)
    plt.yticks(fontsize=32)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='orange', linewidth=3.0)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='blue', linewidth=3.0)
    plt.legend(fontsize=32, loc='lower right')
    plt.title('Accuracy Over Epochs', fontsize=38)
    plt.tight_layout(pad=3.0)
    plt.xlabel("Epochs", fontsize=36)
    plt.ylabel("Accuracy", fontsize=36)
    plt.xticks(ticks=epochs, fontsize=32)
    plt.yticks(fontsize=32)
    
    plt.subplots_adjust(wspace=0.3)

    plt.savefig('loss_acc_plt.png', format='png', dpi=300)
    plt.show()



# INITIALIZATION
# Generating k-mer dictionary
k = 6
kmer_dict = generate_kmer_dict(k)

# Hyperparameters
vocab_size = len(kmer_dict) + 1
d_model = 128
nhead = 8
num_layers = 2
dim_feedforward = 512
num_classes = len(unique_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initializing model
model = GenomeClassifier(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    num_classes=num_classes,
    device=device,
).to(device)

num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Get corresponding class names/labels
classes = label_encoder.classes_
label_mapping = {i:label for i, label in enumerate(classes)}
print("Label Mapping:", label_mapping, '\n')

organism_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
species_names = [taxa_mapping[org] for org in organism_names]

# Splitting into train, test, and validation sets
# 60/20/20 split
train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.2, random_state=42)
train_sequences, val_sequences, train_labels, val_labels = train_test_split(train_sequences, train_labels, test_size=0.25, random_state=42)

# Create the train and validation datasets
train_dataset = SequenceDataset(train_sequences, train_labels, label_encoder)
val_dataset = SequenceDataset(val_sequences, val_labels, label_encoder)
test_dataset = SequenceDataset(test_sequences, test_labels, label_encoder)

# Create the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Early stopping
best_val_loss = float('inf')
patience = 5
epochs_without_improvement = 0


# TRAINING
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=25, device='cuda'):
    global best_val_loss, epochs_without_improvement

    model = model.to(device)

    # Lists to store training and validation metrics for each epoch
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = correct / total

        # Append training metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Reset these lists at the start of each validation phase
        all_val_preds = []
        all_val_labels = []

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Store predictions and true labels
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss / len(val_dataloader)
        val_epoch_acc = correct_val / total_val

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_without_improvement = 0
        else: 
            epochs_without_improvement += 1

        # Append validation metrics
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

	    # Print accuracy and loss
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')
        print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}')

        # Early stopping if no improvement
        if epochs_without_improvement >= patience:
            print(f'Early stopping activated after {epoch} epochs')
            break

    # Print final summary
    print("Number of Epochs:", num_epochs)
    print(f'Final Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}')
    print(f'Final Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}')

    # Classification Report and Confusion Matrix
    print("\nClassification Report:")
    print(classification_report(all_val_labels, all_val_preds, target_names=species_names))
    print('\n')

    # PLOTS
    # Confusion matrix
    plot_confusion_matrix(all_val_labels, all_val_preds, species_names)

    # Loss vs accuracy curve
    plot_loss_accuracy_curve(train_losses, train_accuracies, val_losses, val_accuracies, num_epochs)


train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=num_epochs)


# TESTING
def test_model(model, test_dataloader, criterion, device='cuda'):
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and true labels
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # Calculate and print average test loss and accuracy
    test_loss /= len(test_dataloader)
    accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    # Classification Report and Confusion Matrix
    print("\nClassification Report for Test Data:")
    print(classification_report(all_test_labels, all_test_preds, target_names=species_names))

    # Heatmap plot
    plot_confusion_matrix(all_test_labels, all_test_preds, species_names, is_training=False)


test_model(model, test_dataloader, criterion)



# # Training in chunks
# model_save_dir = "/home/ajcagle/sgtc/model_training"

# if not os.path.exists(model_save_dir):
#     os.makedirs(model_save_dir)

# chunk_epochs = 10   # Number of epochs to train each chunk
# num_chunks = 5      # Number of total chunks

# total_start_time = time.time()

# for chunk_idx in range(num_chunks):
#     chunk_start_time = time.time()

#     print(f"Training on Chunk {chunk_idx + 1}:")

#     sequences = []
#     labels = []

#     for file in files:
#         for subseq, label in generate_random_sample(file, num_subsequences=10000):
#             sequences.append(subseq)
#             labels.append(label)

#     # encoded_labels = label_encoder.transform(chunk_labels)

#     # Shuffle the data
#     sequences, labels = shuffle(sequences, labels, random_state=42)

#     # Split the data
#     sequences_train, sequences_val, labels_train, labels_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

#     # Create the train and validation datasets
#     train_dataset = SequenceDataset(sequences_train, labels_train, label_encoder)
#     val_dataset = SequenceDataset(sequences_val, labels_val, label_encoder)

#     # Create the DataLoaders
#     train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=chunk_epochs)

#     chunk_end_time = time.time()
#     chunk_elapsed_time = chunk_end_time - chunk_start_time
#     total_elapsed_time = chunk_end_time - total_start_time
#     estimated_total_time = (total_elapsed_time / (chunk_idx + 1)) * num_chunks

#     print(f"Chunk {chunk_idx+1} completed in {chunk_elapsed_time:.2f} seconds.")
#     print(f"Total elapsed time: {total_elapsed_time:.2f} seconds.")
#     print(f"Estimated total time for all chunks: {estimated_total_time:.2f} seconds.")

#     # Save the model after each chunk
#     current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
#     save_path = f"{model_save_dir}/model_checkpoint_chunk_{chunk_idx + 1}_{current_time}.pth"
#     torch.save(model.state_dict(), save_path)
#     print(f"Saved model checkpoint for chunk {chunk_idx + 1} at {save_path}")

#     print('\n')

# total_end_time = time.time()
# print(f"Total training time: {total_end_time - total_start_time:.2f} seconds.")