from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt



# ---------------- Hyperparameters & Constants ----------------
DEFAULT_EMBEDDING_DIM = 50
HIDDEN_DIM = 250
CONTEXT_SIZE = 2
PAD_TOKEN = "<PAD>"  # start-of-sentence padding token
UNK_TOKEN = "<UNK>"
WINDOW_SIZE = 2 * CONTEXT_SIZE + 1
FIG_DIR = "figures"
SEED = 42
MASK_UNK_PROB = 0.15
TASK_HYPERPARAMS = {
    "pretrain": {"learning_rate": 1e-3, "epochs": 5, "batch_size": 64},
    "no_pretrain": {"learning_rate": 1e-3, "epochs": 15,  "batch_size": 64},
}
# ------------- Reproducibility / Determinism ----------------

random.seed(SEED)
numpy_seed = SEED
np.random.seed(numpy_seed)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
# Ensure deterministic behavior in cuDNN (may slow down training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# -------------------------------------------------------------

# ---------- Data Processing Functions ----------

def read_labeled_data(path: str) -> List[Tuple[List[str], List[str]]]:
    """
    Read labeled data from file with word/tag pairs.
    
    Args:
        path: Path to file containing labeled data
        
    Returns:
        List of sentences, each containing words and their tags
    """
    sentences = []
    current_words, current_tags = [], []
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Skip document headers
            if line.startswith("-DOCSTART-"):
                continue
                
            # Handle sentence boundaries
            if not line:
                if current_words:
                    sentences.append((current_words, current_tags))
                    current_words, current_tags = [], []
                continue
                
            # Process word/tag pairs
            word, tag = line.split()[:2]
            current_words.append(word)
            current_tags.append(tag)
            
    # Handle last sentence if file doesn't end with empty line
    if current_words:
        sentences.append((current_words, current_tags))
        
    return sentences


def read_unlabeled_data(path: str) -> List[List[str]]:
    """
    Read unlabeled data from file with just words.
    
    Args:
        path: Path to file containing unlabeled data
        
    Returns:
        List of sentences, each containing only words
    """
    sentences = []
    current_words = []
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Skip document headers
            if line.startswith("-DOCSTART-"):
                continue
                
            # Handle sentence boundaries
            if not line:
                if current_words:
                    sentences.append(current_words)
                    current_words = []
                continue
                
            # Process words
            current_words.append(line.split()[0])
            
    # Handle last sentence if file doesn't end with empty line
    if current_words:
        sentences.append(current_words)
        
    return sentences

# ---------- Vocabulary Building Functions ----------

def build_vocab(train_sentences, lowercase=False):
    """
    Build vocabulary from training sentences.
    
    Args:
        train_sentences: List of (words, tags) tuples
        lowercase: Whether to lowercase all words
        
    Returns:
        word2idx: Dictionary mapping words to indices
        idx2word: List mapping indices to words
    """
    word_freq = {}
    
    # Count word frequencies
    for words, _ in train_sentences:
        for word in words:
            word = word.lower() if lowercase else word
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Create vocabulary with special tokens first
    idx2word =   [PAD_TOKEN, UNK_TOKEN] + sorted(word_freq) 
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    
    return word2idx, idx2word


def build_tag_map(train_sentences):
    """
    Build mapping between tags and indices.
    
    Args:
        train_sentences: List of (words, tags) tuples
        
    Returns:
        tag2idx: Dictionary mapping tags to indices
        idx2tag: List mapping indices to tags
    """
    unique_tags = sorted({tag for _, tags in train_sentences for tag in tags})
    tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    
    return tag2idx, unique_tags


def vectorize(sentences, word2idx, tag2idx=None, *, lowercase=False, is_test=False):
    """
    Convert sentences to tensors of word indices and tag indices.
    
    Args:
        sentences: List of sentences (either [words, tags] or just words if is_test=True)
        word2idx: Dictionary mapping words to indices
        tag2idx: Dictionary mapping tags to indices (not used if is_test=True)
        lowercase: Whether to lowercase all words
        is_test: Whether the data is for testing (no tags available)
        
    Returns:
        X: Tensor of word indices
        y: Tensor of tag indices (only if is_test=False)
    """
    pad = CONTEXT_SIZE
    X, y = [], []
    
    for sentence in sentences:
        words = sentence if is_test else sentence[0]
        
        if lowercase:
            words = [w.lower() for w in words]
            
        # Pad sentence with PAD_TOKEN on both sides
        padded = [PAD_TOKEN] * pad + words + [PAD_TOKEN] * pad
        
        # Create windows for each word in the sentence
        for i in range(len(words)):
            window = padded[i:i+WINDOW_SIZE]
            X.append([word2idx.get(w, word2idx[UNK_TOKEN]) for w in window])
            
            if not is_test:
                y.append(tag2idx[sentence[1][i]])
                
    X_tensor = torch.tensor(X, dtype=torch.long)
    
    if is_test:
        return X_tensor
    else:
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_tensor, y_tensor


# ---------- Embedding Functions ----------

def load_pretrained(vec_path: str | Path,
                   vocab_path: str | Path,
                   word2idx: Dict[str, int],
                   lowercase: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Load pretrained embeddings from two-file format.
    
    Args:
        vec_path: Path to vectors file (floats only)
        vocab_path: Path to vocabulary file (one word per line)
        word2idx: Dictionary mapping words to indices
        lowercase: Whether to lowercase words when matching with vocabulary
        
    Returns:
        embedding_matrix: Tensor containing embeddings for words in word2idx
        dimension: Dimensionality of the embeddings
    """
    # Read vocabulary and vectors files
    with open(vocab_path, encoding="utf-8") as f:
        vocab_lines = [line.rstrip() for line in f]
        
    with open(vec_path, encoding="utf-8") as f:
        vec_lines = [line.rstrip() for line in f]
        
    # Validate file lengths
    if len(vocab_lines) != len(vec_lines):
        raise ValueError("Vocabulary and vector files have different lengths")
        
    # Determine embedding dimension from first vector
    dimension = len(vec_lines[0].split())
    
    # Initialize embedding matrix with small random values
    embedding_matrix = torch.randn(len(word2idx), dimension) * 0.1
    
    # Fill embedding matrix with pretrained vectors when available
    found = 0
    for word, vec_str in zip(vocab_lines, vec_lines):
        word = word.lower() if lowercase else word
        
        if word in word2idx:
            vec = torch.tensor(list(map(float, vec_str.split())), dtype=torch.float)
            
            if vec.numel() != dimension:
                raise ValueError("Inconsistent vector dimension in file")
                
            embedding_matrix[word2idx[word]] = vec
            found += 1
            
    print(f"Loaded {found}/{len(word2idx)} pretrained vectors (dim={dimension}).")
    return embedding_matrix, dimension


# ---------- Model Definition ----------

class WindowTagger(nn.Module):
    """
    Neural network model for window-based sequence tagging.
    """
    def __init__(self, 
                vocab_size,
                embedding_dim, 
                context_size, 
                hidden_dim, 
                num_tags, 
                pretrained):
        """
        Initialize the WindowTagger model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            context_size: Number of words on each side of the target word
            hidden_dim: Size of hidden layer
            num_tags: Number of possible tags
            pretrained: Optional pretrained embedding matrix
        """
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with pretrained embeddings if provided
        if pretrained is not None:
            if pretrained.shape != (vocab_size, embedding_dim):
                raise ValueError("Pretrained embedding shape mismatch")
            self.embedding.weight.data.copy_(pretrained)
            
        # Calculate window size
        window_size = 2 * context_size + 1
        
        # Fully connected layers
        self.fc1 = nn.Linear(window_size * embedding_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, num_tags)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of word indices [batch_size, window_size]
            
        Returns:
            Output tensor of tag scores [batch_size, num_tags]
        """
        # Get embeddings and flatten window dimension
        embeddings = self.embedding(x)  # [batch_size, window_size, embedding_dim]
        flat_embeddings = embeddings.view(x.size(0), -1)  # [batch_size, window_size * embedding_dim]
        
        # Pass through layers
        hidden = self.activation(self.fc1(flat_embeddings))
        output = self.fc2(hidden)
        
        return output


# ---------- Training Functions ----------
def run_epoch(model, data_loader, criterion, optimizer=None, task=None):
    """
    Run one epoch of training or evaluation.
    
    Args:
        model: The model to train/evaluate
        data_loader: DataLoader containing batches
        criterion: Loss function
        optimizer: Optimizer for training (None for evaluation)
        task: Task type ('pos' or 'ner') to use appropriate evaluation metrics
        
    Returns:
        average_loss: Average loss per sample
        accuracy: Classification accuracy (modified for NER)
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    # For NER specific evaluation
    ner_correct, ner_total = 0, 0
    
    with torch.set_grad_enabled(is_training):
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # randomly mask 15% of words to learn representation for unknown token
            if is_training:
                mask = torch.rand_like(X_batch, dtype=torch.float) < MASK_UNK_PROB
                X_batch = X_batch.masked_fill(mask, unk_idx)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass (training only)
            if is_training:

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Track standard statistics
            total_loss += loss.item() * X_batch.size(0)
            predictions = outputs.argmax(1)
            correct_predictions += (predictions == y_batch).sum().item()
            total_samples += y_batch.numel()
            
            # NER specific evaluation (ignoring O-O pairs)
            if task == "ner" and not is_training:
                o_tag_idx = tag2idx['O'] 
                non_oo_mask = (y_batch != o_tag_idx) | (predictions != o_tag_idx)
                ner_correct += ((predictions == y_batch) & non_oo_mask).sum().item()
                ner_total += non_oo_mask.sum().item()
            
    average_loss = total_loss / len(data_loader.dataset)
    standard_accuracy = correct_predictions / total_samples

    
    # Return NER-specific accuracy if applicable
    if task == "ner" and not is_training and ner_total > 0:
        ner_accuracy = ner_correct / ner_total
        return average_loss, ner_accuracy
    else:
        return average_loss, standard_accuracy



# ---------- Command Line Interface ----------

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser("Window tagger - minimal CLI")
    parser.add_argument("--task", choices=["pos", "ner"], required=True, 
                      help="Task to perform (part-of-speech tagging or named entity recognition)")
    parser.add_argument("--part", help="name for assignemnt part", required=True)
    parser.add_argument("--vec_path", help="Vectors file (floats only)")
    parser.add_argument("--vocab_path", help="Vocabulary file (one word per line)")
    parser.add_argument("--output_test", help="File for blind-test predictions")

    
    return parser.parse_args()

def process_test_data(task, output_path, model, word2idx, tag2idx, idx2tag):
    """
    Process and save predictions for test data.
    
    Args:
        task: Task name ('pos' or 'ner')
        output_path: Path to save predictions
        model: Trained model
        word2idx: Word to index mapping
        tag2idx: Tag to index mapping
        idx2tag: Index to tag mapping
    """
    print(f"\nProcessing test data and saving to {output_path}...")
    
    # Read test data
    test_path = f"{task}/test"
    if os.path.exists(test_path):
        test_sentences = read_unlabeled_data(test_path)
        print(f"Loaded {len(test_sentences)} test sentences")
        
        # Make predictions
        model.eval()
        all_predictions = []
        
        for sentence in test_sentences:
            # Vectorize the sentence
            X_test = vectorize([sentence], word2idx, is_test=True, lowercase=True)
            X_test = X_test.to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(X_test)
                predictions = outputs.argmax(1).cpu().numpy()
                
            # Map indices back to tags
            predicted_tags = [idx2tag[idx] for idx in predictions]
            all_predictions.append((sentence, predicted_tags))
        
        # Write predictions to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for words, tags in all_predictions:
                for word, tag in zip(words, tags):
                    f.write(f"{word} {tag}\n")
                f.write("\n")  # Empty line between sentences
                
        print(f"Test predictions saved to {output_path}")
    else:
        print(f"Test file not found at {test_path}")

# ---------- Main Function ----------

if __name__ == "__main__":

    """Main function to run the tagger."""
    args = parse_args()
    
    # Set up device (GPU or CPU)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load data
    train_path = f"{args.task}/train"
    dev_path = f"{args.task}/dev"
    test_path = f"{args.task}/test"
    
    train_sentences = read_labeled_data(train_path)
    dev_sentences = read_labeled_data(dev_path)
    
    print(f"Loaded {len(train_sentences)} training sentences")
    print(f"Loaded {len(dev_sentences)} development sentences")
    
    word2idx, idx2word = build_vocab(train_sentences, lowercase=True)

    tag2idx, idx2tag = build_tag_map(train_sentences)
    
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Number of tags: {len(tag2idx)}")
    
    # 3. Load embeddings if specified
    if args.vec_path and args.vocab_path:

        params = TASK_HYPERPARAMS['pretrain']

        pretrained_embeddings, embedding_dim = load_pretrained(
            args.vec_path, args.vocab_path, word2idx, lowercase=True
        )
    elif args.vec_path or args.vocab_path:
        raise ValueError("Need both --vec_path and --vocab_path or neither")
    else:
        embedding_dim, pretrained_embeddings = DEFAULT_EMBEDDING_DIM, None
        print(f"Using random embeddings with dimension {embedding_dim}")
        params =TASK_HYPERPARAMS['no_pretrain']

    X_train, y_train = vectorize(train_sentences, word2idx, tag2idx, lowercase=True)
    X_dev, y_dev = vectorize(dev_sentences, word2idx, tag2idx, lowercase=True)

    train_loader = DataLoader(
    TensorDataset(X_train, y_train), 
    batch_size=params["batch_size"], 
    shuffle=True
    )
    
    dev_loader = DataLoader(
        TensorDataset(X_dev, y_dev), 
        batch_size=params["batch_size"]
    )
        

    # 5. Initialize model
    model = WindowTagger(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        context_size=CONTEXT_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_tags=len(tag2idx),
        pretrained=pretrained_embeddings
    ).to(device)
    
    # Print model structure
    print(f"\nModel architecture:\n{model}\n")
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    
    unk_idx = word2idx[UNK_TOKEN]

    # 6. Training loop
    print("Starting training...")
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Dev Loss':^10} | {'Dev Acc':^10}")
    print("-" * 42)
    
    dev_losses, dev_accuracies = [], []
    os.makedirs(FIG_DIR, exist_ok=True)
    
    for epoch in range(1, params["epochs"] + 1):
        
        # Train and evaluate
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer,task=args.task)


        dev_loss, dev_accuracy = run_epoch(model, dev_loader, criterion, task=args.task)
        
        # Store metrics
        dev_losses.append(dev_loss)
        dev_accuracies.append(dev_accuracy)
        
        # Print progress
        print(f"{epoch:^6d} | {train_loss:^10.3f} | {dev_loss:^10.3f} | {dev_accuracy:^10.3f}")
    
    # 7. Save learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1,  params["epochs"] + 1), dev_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy")
    plt.title(f"{args.task.upper()} - Development Set Accuracy")
    plt.grid(True)
    plt.savefig(f"{FIG_DIR}/part{args.part}_{args.task}_dev_accuracy.png")
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1,  params["epochs"] + 1), dev_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Dev Loss")
    plt.title(f"{args.task.upper()} - Development Set Loss")
    plt.grid(True)
    plt.savefig(f"{FIG_DIR}/part{args.part}_{args.task}_dev_loss.png")
    plt.close()
    
    print(f"\nTraining complete. Plots saved to {FIG_DIR}/")
    
    # 8. Process test data if requested
    if args.output_test:
        process_test_data(args.task, args.output_test, model, word2idx, tag2idx, idx2tag)


