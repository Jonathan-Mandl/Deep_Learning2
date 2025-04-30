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
TASK_HYPERPARAMS = {
    "pos": {"learning_rate": 1e-3, "epochs": 5, "batch_size": 64},
    "ner": {"learning_rate": 1e-3, "epochs": 10,  "batch_size": 64},
}

MASK_UNK_PROB = 0.15
DEFAULT_EMBEDDING_DIM = 50
HIDDEN_DIM = 250
CONTEXT_SIZE = 2
PAD_TOKEN = "<PAD>"  # start-of-sentence padding token
UNK_TOKEN = "<UNK>"
WINDOW_SIZE = 2 * CONTEXT_SIZE + 1
FIG_DIR = "figures"
SEED = 42


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

def build_affix_maps(sentences, affix_len=3):
    """
    Scan all words in train set and collect every unique
    prefix and suffix of length affix_len, reserving 0 for unknown.
    """
    prefixes = set()
    suffixes = set()
    for words, _ in sentences:
        for w in words:
            w_low = w.lower()
            if len(w_low) >= affix_len:
                prefixes.add(w_low[:affix_len])
                suffixes.add(w_low[-affix_len:])
            else:
                # you could also choose to include short words as their own affix
                prefixes.add(w_low)
                suffixes.add(w_low)

    # sort so indices are deterministic
    sorted_pref = sorted(prefixes)
    sorted_suf  = sorted(suffixes)

    # 0 is reserved for any unseen prefix/suffix
    prefix2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, p in enumerate(sorted_pref, start=2):
        prefix2idx[p] = i

    suffix2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, s in enumerate(sorted_suf, start=2):
        suffix2idx[s] = i

    return prefix2idx, suffix2idx


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
    word_freq: Dict[str, int] = {}
    
    # Count word frequencies
    for words, _ in train_sentences:
        for word in words:
            word = word.lower() if lowercase else word
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Create vocabulary with special tokens first
    idx2word = [PAD_TOKEN, UNK_TOKEN] + sorted(word_freq)
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



def vectorize(sentences, word2idx,tag2idx, prefix2idx, suffix2idx, affix_len = 3,lowercase = False,is_test = False):
    """
    Vectorize each sentence into sliding windows of
    (word, prefix, suffix) indices, plus tags if training.
    """
    pad = CONTEXT_SIZE
    window_size = WINDOW_SIZE

    Xw_windows = []
    Xp_windows = []
    Xs_windows = []
    y_labels   = []

    for sent in sentences:
        words = sent if is_test else sent[0]
        tags  = None if is_test else sent[1]

        # build padded index sequences
        #  — words
        word_idxs = [word2idx[PAD_TOKEN]] * pad + [
            word2idx.get(w.lower() if lowercase else w, word2idx[UNK_TOKEN])
            for w in words
        ] + [word2idx[PAD_TOKEN]] * pad

        #  — prefixes
        pref_idxs = [prefix2idx[PAD_TOKEN]] * pad 
        for w in words:
            w_low = w.lower()
            p = w_low[:affix_len] if len(w_low) >= affix_len else w_low
            pref_idxs.append(prefix2idx.get(p,prefix2idx[UNK_TOKEN]))
        pref_idxs += [prefix2idx[PAD_TOKEN]] * pad 

        #  — suffixes
        suf_idxs = [suffix2idx[PAD_TOKEN]] * pad
        for w in words:
            w_low = w.lower()
            s = w_low[-affix_len:] if len(w_low) >= affix_len else w_low
            suf_idxs.append(suffix2idx.get(s, suffix2idx[UNK_TOKEN]))
        suf_idxs += [suffix2idx[PAD_TOKEN]] * pad

        # slide windows
        for i in range(len(words)):
            start = i
            end   = i + window_size
            Xw_windows.append(word_idxs[start:end])
            Xp_windows.append(pref_idxs[start:end])
            Xs_windows.append(suf_idxs[start:end])

            if not is_test:
                y_labels.append(tag2idx[tags[i]])

    # to tensors
    Xw = torch.tensor(Xw_windows, dtype=torch.long)
    Xp = torch.tensor(Xp_windows, dtype=torch.long)
    Xs = torch.tensor(Xs_windows, dtype=torch.long)

    if is_test:
        return Xw, Xp, Xs
    else:
        y = torch.tensor(y_labels, dtype=torch.long)
        return Xw, Xp, Xs, y


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
    Neural network model for window-based sequence tagging with affix embeddings.
    """
    def __init__(self,
                 vocab_size,
                 prefix_vocab_size,
                 suffix_vocab_size,
                 embedding_dim,
                 context_size,
                 hidden_dim,
                 num_tags,
                 pretrained_words=None,
                 pretrained_prefix=None,
                 pretrained_suffix=None):
        super().__init__()
        window_size = 2 * context_size + 1

        # word embeddings
        self.word_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_words is not None:
            self.word_emb.weight.data.copy_(pretrained_words)

        # prefix embeddings
        self.pref_emb = nn.Embedding(prefix_vocab_size, embedding_dim, padding_idx=0)
        if pretrained_prefix is not None:
            self.pref_emb.weight.data.copy_(pretrained_prefix)

        # suffix embeddings
        self.suf_emb = nn.Embedding(suffix_vocab_size, embedding_dim, padding_idx=0)
        if pretrained_suffix is not None:
            self.suf_emb.weight.data.copy_(pretrained_suffix)

        # classifier
        self.fc1 = nn.Linear(window_size * embedding_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, num_tags)

    def forward(self, xw, xp, xs):
        # xw, xp, xs: [batch, window_size]
        ew = self.word_emb(xw)   # [B, W, D]
        ep = self.pref_emb(xp)   # [B, W, D]
        es = self.suf_emb(xs)    # [B, W, D]
        e  = ew + ep + es        # sum elementwise

        flat = e.view(e.size(0), -1)           # [B, W*D]
        h    = self.activation(self.fc1(flat)) # [B, H]
        out  = self.fc2(h)                     # [B, num_tags]
        return out


# ---------- Training Functions ----------
def run_epoch(model, data_loader, criterion, optimizer=None, task=None):
    """
    Run one epoch of training or evaluation.
    
    Works with either:
      - DataLoader yielding (X, y)
      - DataLoader yielding (Xw, Xp, Xs, y)
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    
    total_loss, correct, total = 0.0, 0, 0
    ner_correct, ner_total = 0, 0
    
    with torch.set_grad_enabled(is_training):
        for batch in data_loader:

            Xw, Xp, Xs, y = batch
            Xw, Xp, Xs, y = Xw.to(device), Xp.to(device), Xs.to(device), y.to(device)

            # mask 15% of words to learn representation to UNK token
            if is_training:
                mask = torch.rand_like(Xw, dtype=torch.float) < MASK_UNK_PROB
                Xw = Xw.masked_fill(mask, unk_idx)
                Xs = Xs.masked_fill(mask, unk_idx)
                Xp = Xp.masked_fill(mask, unk_idx)

            outputs = model(Xw, Xp, Xs)

            loss = criterion(outputs, y)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # accumulate
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.numel()
            
            # NER‐specific (ignore O–O)
            if task == "ner" and not is_training:
                o_idx = tag2idx["O"]
                mask = (y != o_idx) | (preds != o_idx)
                ner_correct += ((preds == y) & mask).sum().item()
                ner_total   += mask.sum().item()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    if task == "ner" and not is_training and ner_total > 0:
        return avg_loss, ner_correct / ner_total
    else:
        return avg_loss, accuracy


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
    parser.add_argument("--vec_path", help="Vectors file (floats only)",required=False)
    parser.add_argument("--vocab_path", help="Vocabulary file (one word per line)",required=False)
    parser.add_argument("--output_test", help="File for blind-test predictions",required=False)

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
            Xw_test, Xp_test, Xs_test  = vectorize(
            [sentence],
            word2idx=word2idx,
            tag2idx=tag2idx,
            prefix2idx=prefix2idx,
            suffix2idx=suffix2idx,
            affix_len=3,
            lowercase=True,
            is_test=True
            )
            
            # Vectorize the sentence
            Xw_test, Xp_test, Xs_test = Xw_test.to(device), Xp_test.to(device), Xs_test.to(device)
        
            # Get predictions
            with torch.no_grad():
                outputs = model(Xw_test, Xp_test, Xs_test)
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

    params = TASK_HYPERPARAMS[args.task]

    learning_rate = params["learning_rate"]
    epochs        = params["epochs"]
    batch_size    = params["batch_size"]
    
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

    prefix2idx, suffix2idx = build_affix_maps(train_sentences, affix_len=3)

    word2idx, idx2word = build_vocab(train_sentences, lowercase=True)

    tag2idx, idx2tag = build_tag_map(train_sentences)
    
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Number of tags: {len(tag2idx)}")
    
    # load pretrained embeddings if they are provided
    if args.vec_path and args.vocab_path:
        condition = "pretrained"
        # words
        word_pretrained, embedding_dim = load_pretrained(
            args.vec_path, args.vocab_path, word2idx, lowercase=True
        )
        # prefixes
        pref_pretrained, _ = load_pretrained(
            args.vec_path, args.vocab_path, prefix2idx, lowercase=True
        )
        # suffixes
        suf_pretrained, _ = load_pretrained(
            args.vec_path, args.vocab_path, suffix2idx, lowercase=True
        )
    else:
        condition = "no_pretrained"
        embedding_dim = DEFAULT_EMBEDDING_DIM
        word_pretrained = pref_pretrained = suf_pretrained = None
        print(f"Using random embeddings (dim={embedding_dim}) for all three.")

    # vectorize train sentence to the words, prefixes and suffixes of each window
    Xw_train, Xp_train, Xs_train, y_train = vectorize(
    train_sentences,
    word2idx=word2idx,
    tag2idx=tag2idx,
    prefix2idx=prefix2idx,
    suffix2idx=suffix2idx,
    affix_len=3,
    lowercase=True,
    is_test=False
    )

    # vectorize dev sentence to the words, prefixes and suffixes of each window
    Xw_dev, Xp_dev, Xs_dev, y_dev = vectorize(
        dev_sentences,
        word2idx=word2idx,
        tag2idx=tag2idx,
        prefix2idx=prefix2idx,
        suffix2idx=suffix2idx,
        affix_len=3,
        lowercase=True,
        is_test=False
    )

    train_loader = DataLoader(
    TensorDataset(Xw_train, Xp_train, Xs_train, y_train),
    batch_size=params["batch_size"], shuffle=True
    )
    dev_loader = DataLoader(
        TensorDataset(Xw_dev, Xp_dev, Xs_dev, y_dev),
        batch_size=params["batch_size"]
    )

    # 5. Initialize model
    model = WindowTagger(
    len(word2idx),
    len(prefix2idx),
    len(suffix2idx),
    embedding_dim,
    CONTEXT_SIZE,
    HIDDEN_DIM,
    len(tag2idx),
    pretrained_words=word_pretrained,
    pretrained_prefix=pref_pretrained,
    pretrained_suffix=suf_pretrained
    ).to(device)

    # Print model structure
    print(f"\nModel architecture:\n{model}\n")
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = params["learning_rate"])
    
    pad_idx = word2idx[PAD_TOKEN]          # almost always 0
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
    plt.plot(range(1, params["epochs"] + 1), dev_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy")
    plt.title(f"{args.task.upper()} - Development Set Accuracy")
    plt.grid(True)
    plt.savefig(f"{FIG_DIR}/part4_{condition}_{args.task}_dev_accuracy.png")
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, params["epochs"] + 1), dev_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Dev Loss")
    plt.title(f"{args.task.upper()} - Development Set Loss")
    plt.grid(True)
    plt.savefig(f"{FIG_DIR}/part4_{condition}_{args.task}_dev_loss.png")
    plt.close()
    
    print(f"\nTraining complete. Plots saved to {FIG_DIR}/")
    
    # 8. Process test data if requested
    if args.output_test:
        process_test_data(args.task, args.output_test, model, word2idx, tag2idx, idx2tag)


