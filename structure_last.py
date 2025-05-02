import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt


NUCLEOTIDES = ['A', 'U', 'G', 'C']
NUC_TO_IDX = {nuc: i for i, nuc in enumerate(NUCLEOTIDES)}
VOCAB_SIZE = len(NUCLEOTIDES)  # 4

STRUCT_TOKENS = ['.', '(', ')']
STRUCT_TO_IDX = {tok: i for i, tok in enumerate(STRUCT_TOKENS)}
IDX_TO_STRUCT = {i: tok for tok, i in STRUCT_TO_IDX.items()}
NUM_STRUCT_CLASSES = len(STRUCT_TOKENS)  # 3

class RNAStructureDataset(Dataset):
    def __init__(self, json_file, max_len=None):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        seq = entry["mrna"].replace("T", "U")
        target_struct = entry["mfe"]
        if self.max_len:
            seq = seq[:self.max_len]
            target_struct = target_struct[:self.max_len]
        seq_idx = [NUC_TO_IDX.get(nuc, 0) for nuc in seq] 
        struct_idx = [STRUCT_TO_IDX.get(s, 0) for s in target_struct]
        return torch.tensor(seq_idx, dtype=torch.long), torch.tensor(struct_idx, dtype=torch.long)

def collate_fn(batch):
    seqs, structs = zip(*batch)
    seq_lengths = [len(seq) for seq in seqs]
    max_len = max(seq_lengths)
    
    padded_seqs = torch.zeros(len(seqs), max_len, dtype=torch.long)
    padded_structs = torch.zeros(len(structs), max_len, dtype=torch.long)
    mask = torch.ones(len(seqs), max_len, dtype=torch.bool)
    
    for i, (seq, struct) in enumerate(zip(seqs, structs)):
        length = len(seq)
        padded_seqs[i, :length] = seq
        padded_structs[i, :length] = struct
        mask[i, :length] = False  
    return padded_seqs, padded_structs, mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RNASecondaryStructureModel(nn.Module):
    def __init__(self, input_vocab_size, num_classes, d_model=192, num_layers=12, nhead=12, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, src_key_padding_mask=None):
        embedded = self.embedding(src)
        embedded = embedded.transpose(0, 1) 
        encoded = self.pos_encoder(embedded)
        output = self.transformer_encoder(encoded, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1) 
        logits = self.fc(output)  
        return logits

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for seqs, structs, mask in train_loader:
        seqs, structs, mask = seqs.to(device), structs.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(seqs, src_key_padding_mask=mask)
        logits = logits.view(-1, NUM_STRUCT_CLASSES)
        targets = structs.view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for seqs, structs, mask in test_loader:
            seqs, structs, mask = seqs.to(device), structs.to(device), mask.to(device)
            logits = model(seqs, src_key_padding_mask=mask)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(structs.cpu().tolist())
    return all_preds, all_targets


def idxs_to_string(idxs, idx_to_token):
    if isinstance(idxs[0], list):
        return ["".join([idx_to_token[i] for i in seq]) for seq in idxs]
    else:
        return "".join([idx_to_token[i] for i in idxs])
def evaluate_loss(model, loader, criterion, device):
    """Return mean loss on loader without gradient."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for seqs, structs, mask in loader:
            seqs, structs, mask = seqs.to(device), structs.to(device), mask.to(device)
            logits = model(seqs, src_key_padding_mask=mask)
            logits = logits.view(-1, NUM_STRUCT_CLASSES)
            targets = structs.view(-1)
            total += criterion(logits, targets).item()
    return total / len(loader)
if __name__ == "__main__":
    json_file = "/mnt/disk90/rnadesigndata/uniprot_generated/mfe_results/mfe_results.json"
    dataset = RNAStructureDataset(json_file, max_len=300)

    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    test_split  = int(0.1 * total_size)         
    val_split   = int(0.1 * total_size)         
    test_idx    = indices[:test_split]
    val_idx     = indices[test_split:test_split + val_split]
    train_idx   = indices[test_split + val_split:]

    train_ds, val_ds, test_ds = map(Subset,
                                    (dataset, dataset, dataset),
                                    (train_idx, val_idx, test_idx))

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RNASecondaryStructureModel(
        input_vocab_size=VOCAB_SIZE,
        num_classes=NUM_STRUCT_CLASSES,
        d_model=96, num_layers=8, nhead=8, dropout=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # ---- training loop ----
    best_val_loss = float("inf")
    num_epochs = 2000
    for epoch in range(1, num_epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate_loss(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "structure_best4.pt")

        print(f"Epoch {epoch:4d}/{num_epochs} │ "
              f"train loss: {train_loss:.4f} │ "
              f"val loss: {val_loss:.4f} │ "
              f"best val: {best_val_loss:.4f}")

    print("Training finished. Best model saved to structure_best.pt")
    
    #model.load_state_dict(torch.load("structure_best4.pt"))
    #model.to(device)
   
    test_loss = evaluate_loss(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}")
    
    
    model.eval()
    for seqs, structs, mask in test_loader:
        seqs, structs, mask = seqs.to(device), structs.to(device), mask.to(device)
        #sort sequences by length
        lengths = torch.sum(mask, dim=1)
        lengths, indices = torch.sort(lengths, descending=False)
        seqs = seqs[indices]
        structs = structs[indices]

        with torch.no_grad():
            logits = model(seqs, src_key_padding_mask=mask)
        preds = torch.argmax(logits, dim=-1)
        num_examples = seqs.size(0)
        '''
        for i in range(num_examples):
            # Convert the input sequence (indices) back to RNA letters.
            input_seq = "".join([NUCLEOTIDES[idx] for idx in seqs[i].cpu().tolist() if idx < VOCAB_SIZE])
            # Convert target and predicted structure.
            true_struct = idxs_to_string(structs[i].cpu().tolist(), IDX_TO_STRUCT)
            pred_struct = idxs_to_string(preds[i].cpu().tolist(), IDX_TO_STRUCT)
            print(f"Input RNA: {input_seq}")
            print(f"True Structure: {true_struct}")
            print(f"Predicted Structure: {pred_struct}\n")'''
        #save the first 50 examples to a file
        with open("test_examples2.txt", "w") as f:
            for i in range(num_examples):
                input_seq = "".join([NUCLEOTIDES[idx] for idx in seqs[i].cpu().tolist() if idx < VOCAB_SIZE])
                true_struct = idxs_to_string(structs[i].cpu().tolist(), IDX_TO_STRUCT)
                pred_struct = idxs_to_string(preds[i].cpu().tolist(), IDX_TO_STRUCT)
                f.write(f"Input RNA: {input_seq}\n")
                f.write(f"True Structure: {true_struct}\n")
                f.write(f"Predicted Structure: {pred_struct}\n\n")
        #break
    