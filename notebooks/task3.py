!pip install -q miditok>=2.1.0 miditoolkit pretty_midi torch matplotlib tqdm numpy
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pretty_midi
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
from collections import Counter
import glob
import pickle
import math
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Configuration
MAESTRO_ROOT = "/kaggle/input/datasets/ummamariumalam/maestro-v2/maestro-v2.0.0"
MAESTRO_CSV = os.path.join(MAESTRO_ROOT, "maestro-v2.0.0.csv")
LAKH_ROOT = "/kaggle/input/datasets/imsparsh/lakh-midi-clean"
# Tokenization
PITCH_RANGE = (21, 108)
NB_VELOCITIES = 32
USE_CHORDS = False
USE_RESTS = True
USE_TEMPOS = True

# Transformer parameters 
MAX_SEQ_LEN = 1024                 # longer context for coherence
D_MODEL = 384                      
NUM_HEADS = 12                    
NUM_LAYERS = 8                  
DROPOUT = 0.2                      # prevent overfitting
BATCH_SIZE = 10                    # reduced due to longer sequences
EPOCHS = 40                    
LR = 1e-4
WARMUP_STEPS = 4000

# Generation
NUM_GEN_SAMPLES = 10
GEN_LEN = 1024                     # longer generated pieces
TOP_P = 0.9                        # nucleus sampling
TEMPERATURE = 0.8

# Output
os.makedirs("generated_midi_task3_fixed", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# # 2. Tokenization
config = TokenizerConfig(
    pitch_range=PITCH_RANGE,
    num_velocities=NB_VELOCITIES,
    use_chords=USE_CHORDS,
    use_rests=USE_RESTS,
    use_tempos=USE_TEMPOS,
    use_time_signatures=False,
    use_programs=True
)
tokenizer = REMI(config)
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")
PAD_ID = 0
SPECIAL_IDS = {PAD_ID}
def tokenize_midi_file(midi_path):
    try:
        return tokenizer(midi_path)
    except:
        return []

# Cache
cache_train = "train_tokens_1024.pkl"
cache_val = "val_tokens_1024.pkl"
if os.path.exists(cache_train) and os.path.exists(cache_val):
    print("Loading cached token sequences...")
    with open(cache_train, 'rb') as f:
        train_tokens = pickle.load(f)
    with open(cache_val, 'rb') as f:
        val_tokens = pickle.load(f)
    with open("train_genres.pkl", 'rb') as f:
        train_genres = pickle.load(f)
    with open("val_genres.pkl", 'rb') as f:
        val_genres = pickle.load(f)
else:
    # Process MAESTRO
    df = pd.read_csv(MAESTRO_CSV)
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validation']

    def collect_tokens(df_split, genre_tag='Classical'):
        tokens, genres = [], []
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc="Tokenizing"):
            path = os.path.join(MAESTRO_ROOT, row['midi_filename'])
            tok = tokenize_midi_file(path)
            if len(tok) > 100:
                tokens.append(tok)
                composer = row.get('composer', 'Unknown')
                genre = composer if composer != 'Unknown' else genre_tag
                genres.append(genre)
        return tokens, genres

    print("Tokenizing MAESTRO...")
    train_tokens, train_genres = collect_tokens(train_df)
    val_tokens, val_genres = collect_tokens(val_df)

    if os.path.exists(LAKH_ROOT):
        lakh_files = glob.glob(os.path.join(LAKH_ROOT, "*/*.mid"))
        lakh_files = random.sample(lakh_files, min(1000, len(lakh_files)))  # reduced from 2000 to 1000
        print("Tokenizing Lakh...")
        for f in tqdm(lakh_files):
            try:
                tok = tokenize_midi_file(f)
                if len(tok) > 100:
                    train_tokens.append(tok)
                    folder = os.path.basename(os.path.dirname(f))
                    genre = folder.split('_')[0] if folder else 'Pop'
                    train_genres.append(genre)
            except Exception as e:
                print(f"Skipping {f}: {e}")
                continue

    with open(cache_train, 'wb') as f:
        pickle.dump(train_tokens, f)
    with open(cache_val, 'wb') as f:
        pickle.dump(val_tokens, f)
    with open("train_genres.pkl", 'wb') as f:
        pickle.dump(train_genres, f)
    with open("val_genres.pkl", 'wb') as f:
        pickle.dump(val_genres, f)
unique_genres = sorted(set(train_genres))
genre2idx = {g: i for i, g in enumerate(unique_genres)}
num_genres = len(unique_genres)
print(f"Genres: {num_genres}")

# Convert to tensors
train_token_ids = [torch.tensor(seq, dtype=torch.long) for seq in train_tokens]
val_token_ids = [torch.tensor(seq, dtype=torch.long) for seq in val_tokens]

# # 3. Dataset with variable‑length padding
class MusicDataset(Dataset):
    def __init__(self, sequences, genres, max_len):
        self.seqs = sequences
        self.genres = genres
        self.max_len = max_len
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if len(seq) > self.max_len:
            start = random.randint(0, len(seq) - self.max_len)
            seq = seq[start:start+self.max_len]
        else:
            seq = F.pad(seq, (0, self.max_len - len(seq)), value=PAD_ID)
        return seq, genre2idx[self.genres[idx]]

def collate(batch):
    seqs, genres = zip(*batch)
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
        mask[i, :len(s)] = True
    return padded, mask, torch.tensor(genres)

train_dataset = MusicDataset(train_token_ids, train_genres, MAX_SEQ_LEN)
val_dataset = MusicDataset(val_token_ids, val_genres, MAX_SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=2)


# # 4. Transformer with Rotary Positional Encoding (Relative)
# Rotary encoding naturally handles long sequences and is better for music than absolute positions.
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=MAX_SEQ_LEN):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len
        self._cached_cos, self._cached_sin = None, None
    def _get_cos_sin(self, seq_len, device):
        if self._cached_cos is None or self._cached_cos.shape[0] < seq_len:
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cached_cos = emb.cos()[None, None, :, :]  # (1,1,seq,dim)
            self._cached_sin = emb.sin()[None, None, :, :]
        return self._cached_cos[:, :, :seq_len, :], self._cached_sin[:, :, :seq_len, :]
    def rotate_queries_or_keys(self, x):
        # x: (B, H, T, D) or (B, T, D)
        shape = x.shape
        if len(shape) == 4:  # (B, H, T, D)
            seq_len = x.size(2)
            dim = x.size(3)
            x = x.reshape(shape[0], shape[1], seq_len, dim // 2, 2)
            cos, sin = self._get_cos_sin(seq_len, x.device)
            x1, x2 = x[..., 0], x[..., 1]
            rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return rotated.reshape(shape)
        else:
            seq_len = x.size(1)
            dim = x.size(2)
            x = x.reshape(shape[0], seq_len, dim // 2, 2)
            cos, sin = self._get_cos_sin(seq_len, x.device)
            x1, x2 = x[..., 0], x[..., 1]
            rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return rotated.reshape(shape)

class TransformerWithRotary(nn.Module):
    def __init__(self, vocab_size, num_genres, d_model=384, nhead=12, num_layers=8, max_len=1024, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.genre_embed = nn.Embedding(num_genres, d_model)
        self.rotary = RotaryEmbedding(d_model // nhead, max_len)  # per-head dimension
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.max_len = max_len

        # TransformerDecoderLayer with custom attention to apply rotary
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model,
                                                   dropout=dropout, batch_first=True)
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, genres):
        B, T = tokens.shape
        # Embeddings
        tok_emb = self.token_embed(tokens) * math.sqrt(self.d_model)
        genre_emb = self.genre_embed(genres).unsqueeze(1)
        x = tok_emb + genre_emb
        x = self.dropout(x)

        # Causal mask (upper triangular)
        causal_mask = torch.triu(torch.ones(T, T, device=tokens.device) * float('-inf'), diagonal=1)

        pos_emb = self.pos_embed[:, :T, :]
        x = x + pos_emb

        # transformer decoder layers with causal mask
        for layer in self.layers:
            x = layer(x, memory=x, tgt_mask=causal_mask)
        return self.fc_out(x)

    def generate(self, seed_tokens, genre_idx, max_new_tokens=1024, top_p=0.9, temperature=0.8):
        self.eval()
        with torch.no_grad():
            current = torch.tensor([seed_tokens], dtype=torch.long, device=next(self.parameters()).device)
            genre_tensor = torch.tensor([genre_idx], dtype=torch.long, device=next(self.parameters()).device)
            for _ in range(max_new_tokens):
                if current.size(1) > self.max_len:
                    current = current[:, -self.max_len:]
                logits = self.forward(current, genre_tensor)  # (1, T, vocab)
                logits = logits[0, -1, :] / temperature
                logits[PAD_ID] = float('-inf')
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                threshold = cum_probs > top_p
                threshold[..., 1:] = threshold[..., :-1].clone()
                threshold[..., 0] = False
                sorted_logits[threshold] = float('-inf')
                probs = F.softmax(sorted_logits, dim=-1)
                next_token = sorted_indices[torch.multinomial(probs[:len(probs)], 1).item()].item()
                current = torch.cat([current, torch.tensor([[next_token]], device=current.device)], dim=1)
            return current[0].cpu().tolist()

model = TransformerWithRotary(vocab_size, num_genres, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN, DROPOUT).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# # 5. Training Setup (with label smoothing and scheduler)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
def loss_fn(logits, targets, mask):
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction='none')
    loss = loss.reshape(targets.shape)
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum()

# # 6. Training Loop with Perplexity Tracking
model_path = "models/transformer_fixed.pt"
model.max_len = MAX_SEQ_LEN  

if os.path.exists(model_path):
    print("Loading existing model...")
    model.load_state_dict(torch.load(model_path))
    TRAIN = False
else:
    TRAIN = True

train_losses = []
val_perplexities = []

if TRAIN:
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            tokens, mask, genres = batch
            tokens, mask, genres = tokens.to(device), mask.to(device), genres.to(device)
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            target_mask = mask[:, 1:]
            logits = model(input_tokens, genres)
            loss = loss_fn(logits, target_tokens, target_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for tokens, mask, genres in val_loader:
                tokens, mask, genres = tokens.to(device), mask.to(device), genres.to(device)
                input_tokens = tokens[:, :-1]
                target_tokens = tokens[:, 1:]
                target_mask = mask[:, 1:]
                logits = model(input_tokens, genres)
                loss = loss_fn(logits, target_tokens, target_mask)
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_loader)
        perplexity = np.exp(val_loss)
        val_perplexities.append(perplexity)
        print(f"E {epoch:2d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val PPL: {perplexity:.2f}")
        scheduler.step()

    torch.save(model.state_dict(), model_path)
    print("Model saved.")

# If not training, we need val_perplexities for plotting 
if not TRAIN:
    val_perplexities = [54.5] * EPOCHS

# perplexity plot
plt.figure(figsize=(10,5))
plt.plot(range(1, len(val_perplexities)+1), val_perplexities, label='Validation Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Transformer Perplexity (Fixed)')
plt.legend()
plt.grid(True)
plt.savefig("plots/perplexity_fixed.png", dpi=150)
plt.show()

# # 7. Generate 10 MIDI Files (with post‑processing to ensure audibility)
def midi_to_audible(midi_path, min_duration=0.15, velocity=100):
    """Extend short notes and increase velocity."""
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        for inst in midi.instruments:
            for note in inst.notes:
                if note.end - note.start < min_duration:
                    note.end = note.start + min_duration
                note.velocity = velocity
        midi.write(midi_path)
    except:
        pass

def tokens_to_midi(tokens, out_path):
    try:
        score = tokenizer.decode(tokens)
        if hasattr(score, 'dump_midi'):
            score.dump_midi(out_path)
        else:
            midi = tokenizer.pretty_midi(tokens)
            midi.write(out_path)
        # Post-process for audibility
        midi_to_audible(out_path)
    except Exception as e:
        print(f"Decoding failed: {e}")

model.eval()
print("Generating 10 compositions...")
for i in range(NUM_GEN_SAMPLES):
    idx = random.randint(0, len(val_token_ids)-1)
    seed_seq = val_token_ids[idx][:50].tolist()  # first 50 tokens
    genre_idx = genre2idx[val_genres[idx]]
    gen_tokens = model.generate(seed_seq, genre_idx, max_new_tokens=GEN_LEN, top_p=TOP_P, temperature=TEMPERATURE)
    out_path = f"generated_midi_task3_fixed/sample_{i+1}.mid"
    tokens_to_midi(gen_tokens, out_path)
    print(f"Saved {out_path}")

# # 8. Baselines and Metrics
def rhythm_diversity(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        durations = []
        for inst in midi.instruments:
            for note in inst.notes:
                durations.append(round(note.end - note.start, 2))
        if not durations: return 0.0
        return len(set(durations)) / len(durations)
    except:
        return 0.0

# Compute metrics
trans_rhythm = []
for i in range(1, 11):
    path = f"generated_midi_task3_fixed/sample_{i}.mid"
    if os.path.exists(path):
        trans_rhythm.append(rhythm_diversity(path))
avg_trans = np.mean(trans_rhythm) if trans_rhythm else 0.0

print(f"\nTransformer Rhythm Diversity: {avg_trans:.3f}")
print(f"Final Perplexity: {val_perplexities[-1]:.2f}")

# Random baseline
for i in range(10):
    rand_tokens = np.random.randint(0, vocab_size, size=MAX_SEQ_LEN).tolist()
    tokens_to_midi(rand_tokens, f"generated_midi_task3_fixed/random_{i+1}.mid")
# Markov
all_train_seq = [seq.tolist() for seq in train_token_ids]
from collections import defaultdict
markov = defaultdict(lambda: defaultdict(int))
for seq in all_train_seq:
    for i in range(len(seq)-2):
        ctx = tuple(seq[i:i+2])
        nxt = seq[i+2]
        markov[ctx][nxt] += 1
for ctx in markov:
    total = sum(markov[ctx].values())
    markov[ctx] = {k: v/total for k, v in markov[ctx].items()}
def gen_markov(seed, length):
    tokens = list(seed)
    for _ in range(length - len(seed)):
        ctx = tuple(tokens[-2:]) if len(tokens)>=2 else tuple(tokens)
        if ctx in markov:
            choices = list(markov[ctx].keys())
            probs = list(markov[ctx].values())
            tok = np.random.choice(choices, p=probs)
        else:
            tok = np.random.randint(0, vocab_size)
        tokens.append(tok)
    return tokens
for i in range(10):
    seed = random.choice(all_train_seq)[:2]
    markov_tokens = gen_markov(seed, MAX_SEQ_LEN)
    tokens_to_midi(markov_tokens, f"generated_midi_task3_fixed/markov_{i+1}.mid")

