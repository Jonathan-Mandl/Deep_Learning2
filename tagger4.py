import argparse
import os
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict, Counter

# ---------------- Hyperparameters & Constants ----------------
TASK_HYPERPARAMS = {
    "pos": {"learning_rate": 1e-3, "epochs": 3, "batch_size": 64},
    "ner": {"learning_rate": 1e-3, "epochs": 7,  "batch_size": 64},
}

DEFAULT_EMBEDDING_DIM = 50
MASK_UNK_PROB = 0.15
HIDDEN_DIM = 250
CONTEXT_SIZE = 2
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
WINDOW_SIZE = 2 * CONTEXT_SIZE + 1
CHAR_DIM = 30
CHAR_KERNEL_SIZE = 3
CHAR_PAD = 2
CHAR_FILTERS = 30
RAW_CHAR_LEN = 20
MAX_CHAR_LEN = RAW_CHAR_LEN + 2 * CHAR_PAD
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

def read_labeled_data(path):
    sentences = []
    words = []
    tags = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if not line:
                if words:
                    sentences.append((words, tags))
                    words = []
                    tags = []
                continue
            w, t = line.split()[:2]
            words.append(w)
            tags.append(t)
    if words:
        sentences.append((words, tags))
    return sentences


def read_unlabeled_data(path):
    sentences = []
    words = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if not line:
                if words:
                    sentences.append(words)
                    words = []
                continue
            words.append(line.split()[0])
    if words:
        sentences.append(words)
    return sentences


def build_vocab(train_sentences, lowercase=True):
    freq = {}
    for words, _ in train_sentences:
        for w in words:
            w_l = w.lower() if lowercase else w
            freq[w_l] = freq.get(w_l, 0) + 1
    idx2word = [PAD_TOKEN, UNK_TOKEN] + sorted(freq)
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def build_char_vocab(train_sentences):
    chars = set()
    for words, _ in train_sentences:
        for w in words:
            for c in w:
                chars.add(c)
    idx2char = ['<PAD>', '<UNK>'] + sorted(chars)
    char2idx = {c: i for i, c in enumerate(idx2char)}
    return char2idx, idx2char


def build_tag_map(train_sentences):
    tag_set = set()
    for _, tags in train_sentences:
        for t in tags:
            tag_set.add(t)
    idx2tag = sorted(tag_set)
    tag2idx = {t: i for i, t in enumerate(idx2tag)}
    return tag2idx, idx2tag


def vectorize_with_chars(sentences, word2idx, tag2idx=None, char2idx=None, lowercase=True, is_test=False):
    pad = CONTEXT_SIZE
    Xw, Xc, ys = [], [], []
    for sent in sentences:
        if is_test:
            words, tags = sent, None
        else:
            words, tags = sent
        if lowercase:
            words = [w.lower() for w in words]
        padded = [PAD_TOKEN]*pad + words + [PAD_TOKEN]*pad
        for i in range(len(words)):
            window = padded[i:i+WINDOW_SIZE]
            Xw.append([word2idx.get(w_, word2idx[UNK_TOKEN]) for w_ in window])
            char_mat = []
            for w_ in window:
                raw = list(w_[:RAW_CHAR_LEN])
                padded_chars = ['<PAD>']*CHAR_PAD + raw + ['<PAD>']*CHAR_PAD
                extra = MAX_CHAR_LEN - len(padded_chars)
                if extra>0:
                    left = extra//2; right=extra-left
                    padded_chars = ['<PAD>']*left + padded_chars + ['<PAD>']*right
                char_mat.append([char2idx.get(c,char2idx['<UNK>']) for c in padded_chars[:MAX_CHAR_LEN]])
            Xc.append(char_mat)
            if not is_test:
                ys.append(tag2idx[tags[i]])
    Xw_t = torch.tensor(Xw,dtype=torch.long)
    Xc_t = torch.tensor(Xc,dtype=torch.long)
    if is_test:
        return Xw_t, Xc_t
    return Xw_t, Xc_t, torch.tensor(ys,dtype=torch.long)


def load_pretrained(vec_path, vocab_path, word2idx, lowercase=True):
    with open(vocab_path,encoding="utf-8") as f:
        vs=[l.rstrip() for l in f]
    with open(vec_path,encoding="utf-8") as f:
        ls=[l.rstrip() for l in f]
    if len(vs)!=len(ls): raise ValueError("Mismatch lengths")
    dim=len(ls[0].split())
    M=torch.randn(len(word2idx),dim)*0.1
    found=0
    for w,v in zip(vs,ls):
        wl=w.lower() if lowercase else w
        if wl in word2idx:
            M[word2idx[wl]]=torch.tensor(list(map(float,v.split())))
            found+=1
    print(f"Loaded {found}/{len(word2idx)} vectors dim={dim}")
    return M,dim

class WindowTaggerCharCNN(nn.Module):
    def __init__(self,vocab_size,emb_dim,char_vocab_size,char_dim=CHAR_DIM,
                 char_k=CHAR_KERNEL_SIZE,char_f=CHAR_FILTERS,
                 context_size=CONTEXT_SIZE,hidden_dim=HIDDEN_DIM,
                 num_tags=0,pretrained=None):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)
        self.char_embedding=nn.Embedding(char_vocab_size,char_dim,padding_idx=0)
        b=math.sqrt(3/char_dim)
        nn.init.uniform_(self.char_embedding.weight,-b,b)
        self.char_cnn=nn.Conv1d(char_dim,char_f,kernel_size=char_k)
        win=2*context_size+1
        self.fc1=nn.Linear(win*(emb_dim+char_f),hidden_dim)
        self.act=nn.Tanh()
        self.fc2=nn.Linear(hidden_dim,num_tags)

    def forward(self,xw,xc):
        B,W=xw.size()
        we=self.embedding(xw).view(B,-1)
        ce=self.char_embedding(xc).permute(0,1,3,2)
        B2,W2,C,L=ce.size()
        ce=ce.reshape(B2*W2,C,L)
        ce=self.char_cnn(ce)
        ce, _=torch.max(ce,dim=2)
        ce=ce.view(B,W,-1).reshape(B,-1)
        x=torch.cat([we,ce],dim=1)
        h=self.act(self.fc1(x))
        return self.fc2(h)

def run_epoch(model,loader,criterion,unk_idx,optimizer=None,task=None):
    train=optimizer is not None
    if train: model.train()
    else:     model.eval()
    tot,cor,all=0.0,0,0
    ner_c,ner_t=0,0

    with torch.set_grad_enabled(train):
        for Xw, Xc, y in loader:

            Xw,Xc,y=Xw.to(device),Xc.to(device),y.to(device)

            # mask 15% of words to learn representation to UNK token
            if train:
                mask = torch.rand_like(Xw, dtype=torch.float) < MASK_UNK_PROB
                Xw = Xw.masked_fill(mask, unk_idx)

            out=model(Xw,Xc)
            loss=criterion(out,y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds=out.argmax(1)
            tot+=loss.item()*Xw.size(0)
            cor+= (preds==y).sum().item()
            all+= y.size(0)

            if task=="ner" and not train:
                oi=tag2idx['O']
                mask=(y!=oi)|(preds!=oi)
                ner_c+=((preds==y) & mask).sum().item()
                ner_t += mask.sum().item()
    avg=tot/len(loader.dataset)
    if task=="ner" and not train and ner_t>0: return avg, ner_c/ner_t
    return avg, cor/all

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["pos","ner"],required=True)
    p.add_argument("--vec_path", required =True)
    p.add_argument("--vocab_path", required=True)
    p.add_argument("--output_test")
    return p.parse_args()

def process_test_data(task,output_path,model,word2idx,tag2idx,idx2tag,char2idx):
    s=read_unlabeled_data(f"{task}/test")
    model.eval()
    with open(output_path,'w',encoding='utf-8') as f:
        for sent in s:
            Xw,Xc=vectorize_with_chars([sent],word2idx,None,char2idx,True,True)
            Xw,Xc=Xw.to(device),Xc.to(device)
            with torch.no_grad():
                out=model(Xw,Xc)
            for w,i in zip(sent,out.argmax(1).cpu()): f.write(f"{w} {idx2tag[i]}\n")
            f.write("\n")
    print(f"Saved test to {output_path}")

if __name__=="__main__":

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args=parse_args()

    params = TASK_HYPERPARAMS[args.task]

    learning_rate = params["learning_rate"]
    epochs        = params["epochs"]
    batch_size    = params["batch_size"]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=read_labeled_data(f"{args.task}/train")
    dev=read_labeled_data(f"{args.task}/dev")

    word2idx,idx2w=build_vocab(train,True)
    tag2idx,idx2t=build_tag_map(train)
    char2idx,idx2c=build_char_vocab(train)

    pretrained_embd,embd_dim=load_pretrained(args.vec_path,args.vocab_path,word2idx,True)

    Xw_tr,Xc_tr,y_tr=vectorize_with_chars(train,word2idx,tag2idx,char2idx,True,False)
    Xw_dev,Xc_dev,y_dev=vectorize_with_chars(dev,word2idx,tag2idx,char2idx,True,False)

    tr_ld=DataLoader(TensorDataset(Xw_tr,Xc_tr,y_tr),batch_size=batch_size,shuffle=True)
    dv_ld=DataLoader(TensorDataset(Xw_dev,Xc_dev,y_dev),batch_size=batch_size)

    model=WindowTaggerCharCNN(len(word2idx),embd_dim,len(char2idx),CHAR_DIM,CHAR_KERNEL_SIZE,CHAR_FILTERS,CONTEXT_SIZE,HIDDEN_DIM,len(tag2idx),pretrained_embd).to(device)

    crit=nn.CrossEntropyLoss()
    opt=optim.Adam(model.parameters(),lr=learning_rate)
    os.makedirs(FIG_DIR,exist_ok=True)

    unk_idx = word2idx[UNK_TOKEN]

    for ep in range(1,epochs+1):
        tl,_=run_epoch(model,tr_ld,crit,unk_idx,opt,args.task)
        vl,va=run_epoch(model,dv_ld,crit,unk_idx,None,args.task)
        print(f"Epoch {ep}|Train {tl:.3f}|DevLoss {vl:.3f}|DevAcc {va:.3f}")

    plt.figure()
    plt.plot(range(1,epochs+1),[run_epoch(model,dv_ld,crit,unk_idx,None,args.task)[1] for _ in range(epochs)])
    plt.xlabel("Epoch")
    plt.ylabel("DevAcc")
    plt.savefig(f"{FIG_DIR}/part4_{args.task}_dev_acc.png")
    plt.close()
    plt.figure()
    plt.plot(range(1,epochs+1),[run_epoch(model,dv_ld,crit,unk_idx,None,args.task)[0] for _ in range(epochs)])
    plt.xlabel("Epoch");plt.ylabel("DevLoss")
    plt.savefig(f"{FIG_DIR}/part4_{args.task}_dev_loss.png")
    plt.close()
    if args.output_test: 
        process_test_data(args.task,args.output_test,model,word2idx,tag2idx,idx2t,char2idx)
