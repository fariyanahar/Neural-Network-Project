%pip install -q miditok>=2.1.0 miditoolkit pretty_midi torch matplotlib tqdm numpy
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
import glob
import pickle
import math
import warnings
warnings.filterwarnings('ignore')

# Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Task 3 Model and Tokenizer Load
import os
local_path = "models/transformer_fixed.pt"
input_path = "/kaggle/input/notebooks/ummamariumalam/task3-new"
if os.path.exists(local_path):
    model_path = local_path
elif os.path.exists(input_path):
    model_path = os.path.join(input_path, "models/transformer_fixed.pt")
else:
    raise FileNotFoundError("Model not found")

# Load validation token sequences (to obtain genre mapping and seed sequences)
cache_val_local = "val_tokens_1024.pkl"
cache_val_input = "/kaggle/input/notebooks/ummamariumalam/task3-new/val_tokens_1024.pkl"  # adjust name
genres_local = "val_genres.pkl"
genres_input = "/kaggle/input/notebooks/ummamariumalam/task3-new/val_genres.pkl"
train_genres_local = "train_genres.pkl"
train_genres_input = "/kaggle/input/notebooks/ummamariumalam/task3-new/train_genres.pkl"

# MAESTRO_ROOT = "/kaggle/input/datasets/ummamariumalam/maestro-v2/maestro-v2.0.0"
# MAESTRO_CSV = os.path.join(MAESTRO_ROOT, "maestro-v2.0.0.csv")
# LAKH_ROOT = "/kaggle/input/datasets/imsparsh/lakh-midi-clean"- we dont use this, as task3 outputs are taken as input

# Tokenization parameters same as Task 3
PITCH_RANGE = (21, 108)
NB_VELOCITIES = 32
USE_CHORDS = False
USE_RESTS = True
USE_TEMPOS = True

# Transformer parameters same as Task 3
MAX_SEQ_LEN = 1024
D_MODEL = 384
NUM_HEADS = 12
NUM_LAYERS = 8
DROPOUT = 0.2
BATCH_SIZE = 10        
EPOCHS = 40
LR = 1e-4
WARMUP_STEPS = 4000

# Generation for RL
NUM_RL_ITERATIONS = 10           # policy gradient steps
BATCH_SIZE_RL = 8                # number of samples per iteration
GEN_LEN = 512                    # shorter for faster RL
TOP_P = 0.9
TEMPERATURE = 0.8
KL_WEIGHT = 0.1                  # λ for KL penalty
RL_LR = 1e-5                     # small learning rate for fine‑tuning

# Output
os.makedirs("generated_midi_task4", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# same config as Task 3 to initialize tokenizer 
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
PAD_ID = 0
print(f"Vocabulary size: {vocab_size}")\

# Helper functions for MIDI conversion (must be defined early)
def midi_to_audible(midi_path, min_duration=0.15, velocity=100):
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
        midi_to_audible(out_path)
    except Exception as e:
        print(f"Decoding failed: {e}")

# Load validation token sequences from the input notebook
if os.path.exists(cache_val_local):
    with open(cache_val_local, 'rb') as f:
        val_tokens = pickle.load(f)
    with open(genres_local, 'rb') as f:
        val_genres = pickle.load(f)
elif os.path.exists(cache_val_input):
    with open(cache_val_input, 'rb') as f:
        val_tokens = pickle.load(f)
    with open(genres_input, 'rb') as f:
        val_genres = pickle.load(f)
else:
    raise FileNotFoundError("Pre‑tokenized validation data not found.")
    
if os.path.exists(train_genres_local):
    with open(train_genres_local, 'rb') as f:
        train_genres = pickle.load(f)
elif os.path.exists(train_genres_input):
    with open(train_genres_input, 'rb') as f:
        train_genres = pickle.load(f)
else:
    train_genres = val_genres
unique_genres = sorted(set(train_genres))
genre2idx = {g: i for i, g in enumerate(unique_genres)}
num_genres = len(unique_genres)
print(f"Genres: {num_genres}")

# Convert to tensors (for seed selection)
val_token_ids = [torch.tensor(seq, dtype=torch.long) for seq in val_tokens]

#same Transformer architecture as Task 3
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=MAX_SEQ_LEN):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len
        self._cached_cos, self._cached_sin = None, None

class TransformerWithRotary(nn.Module):
    def __init__(self, vocab_size, num_genres, d_model=384, nhead=12, num_layers=8, max_len=1024, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.genre_embed = nn.Embedding(num_genres, d_model)
        self.rotary = RotaryEmbedding(d_model // nhead, max_len)
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model,
                                                   dropout=dropout, batch_first=True)
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, genres):
        B, T = tokens.shape
        tok_emb = self.token_embed(tokens) * math.sqrt(self.d_model)
        genre_emb = self.genre_embed(genres).unsqueeze(1)
        x = tok_emb + genre_emb
        x = self.dropout(x)
        causal_mask = torch.triu(torch.ones(T, T, device=tokens.device) * float('-inf'), diagonal=1)
        pos_emb = self.pos_embed[:, :T, :]
        x = x + pos_emb
        for layer in self.layers:
            x = layer(x, memory=x, tgt_mask=causal_mask)
        return self.fc_out(x)

    def generate(self, seed_tokens, genre_idx, max_new_tokens=512, top_p=0.9, temperature=0.8):
        self.eval()
        with torch.no_grad():
            current = torch.tensor([seed_tokens], dtype=torch.long, device=next(self.parameters()).device)
            genre_tensor = torch.tensor([genre_idx], dtype=torch.long, device=next(self.parameters()).device)
            for _ in range(max_new_tokens):
                if current.size(1) > self.max_len:
                    current = current[:, -self.max_len:]
                logits = self.forward(current, genre_tensor)
                logits = logits[0, -1, :] / temperature
                logits[PAD_ID] = float('-inf')   # mask pad
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

    def log_prob(self, tokens, genres):
        self.eval()
        with torch.no_grad():
            # tokens: (1, T) tensor
            # genres: (1,) tensor
            logits = self.forward(tokens, genres)  # (1, T, vocab)
            log_probs = F.log_softmax(logits, dim=-1)
            # Shift: target tokens are tokens[:, 1:]
            log_lik = log_probs[:, :-1, :].gather(2, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
            return log_lik.sum(dim=1).item()

# Instantiation of model and loading pretrained weights from Task 3
model = TransformerWithRotary(vocab_size, num_genres, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN, DROPOUT).to(device)
# Use the model_path already determined from the detection block
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded Task 3 model from {model_path}")
else:
    raise FileNotFoundError(f"Task 3 model not found at {model_path}")  # set to train mode for RL updates

original_model = TransformerWithRotary(vocab_size, num_genres, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN, DROPOUT).to(device)
original_model.load_state_dict(torch.load(model_path, map_location=device))
original_model.eval()
for param in original_model.parameters():
    param.requires_grad = False



# 2. Reward Model

# (We train a small neural network on musical features extracted from generated MIDI files.In a real project, you would collect human ratings. Here we simulate by assigning:
# high reward for high rhythm diversity, many notes, and even pitch distribution.
# The reward model will be trained on a small set of generated samples with simulated scores. Afterwards, it will be used to evaluate all RL samples.

# Function to extract features from a MIDI file (piano‑roll representation)
def extract_midi_features(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for inst in midi.instruments:
            for note in inst.notes:
                notes.append((note.pitch, note.start, note.end, note.velocity))
        if not notes:
            return np.zeros(4)
        # Features: number of notes, rhythm diversity, pitch histogram entropy, mean velocity
        durations = [n[2]-n[1] for n in notes]
        pitches = [n[0] for n in notes]
        velocities = [n[3] for n in notes]
        n_notes = len(notes)
        # Rhythm diversity: unique quantized durations / total
        unique_durs = len(set([round(d,2) for d in durations]))
        rhythm_div = unique_durs / n_notes if n_notes>0 else 0
        # Pitch histogram (12 pitch classes)
        pitch_class_counts = [0]*12
        for p in pitches:
            pitch_class_counts[p % 12] += 1
        pitch_hist = np.array(pitch_class_counts) / n_notes
        entropy = -np.sum(pitch_hist * np.log2(pitch_hist+1e-6))
        # Mean velocity
        mean_vel = np.mean(velocities) if velocities else 0
        return np.array([n_notes, rhythm_div, entropy, mean_vel])
    except:
        return np.zeros(4)

# we assign synthetic human scores to simulate a dataset of 100 MIDI files (generate from base model)  
# Score formula: high notes + high rhythm diversity + high entropy + high velocity
def synthetic_score(features):
    n_notes, rhythm_div, entropy, vel = features
    score = (min(n_notes/200,1.0))*2 + rhythm_div*2 + (entropy/3.5)*2 + (vel/100)*1
    return np.clip(score, 1, 5)

# Generation of a set of samples from base model to create training data for reward model
print("Generating synthetic training data for reward model...")
base_model = model
base_model.eval()
features_list = []
scores_list = []
for i in range(100):
    idx = random.randint(0, len(val_token_ids)-1)
    seed_seq = val_token_ids[idx][:50].tolist()
    genre_idx = genre2idx[val_genres[idx]]
    gen_tokens = base_model.generate(seed_seq, genre_idx, max_new_tokens=GEN_LEN, top_p=TOP_P, temperature=TEMPERATURE)
    tmp_path = f"tmp_reward_{i}.mid"
    # Use the global tokens_to_midi (includes post‑processing)
    tokens_to_midi(gen_tokens, tmp_path)
    feats = extract_midi_features(tmp_path)
    score = synthetic_score(feats)
    features_list.append(feats)
    scores_list.append(score)
    os.remove(tmp_path)
features_array = np.array(features_list)
scores_array = np.array(scores_list)
print("Reward model training data ready.")

# reward model (MLP)
class RewardModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

reward_model = RewardModel().to(device)
# Training the reward model on the synthetic data
optimizer_rm = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
criterion_rm = nn.MSELoss()
features_tensor = torch.tensor(features_array, dtype=torch.float32).to(device)
scores_tensor = torch.tensor(scores_array, dtype=torch.float32).unsqueeze(1).to(device)
dataset_rm = torch.utils.data.TensorDataset(features_tensor, scores_tensor)
loader_rm = DataLoader(dataset_rm, batch_size=16, shuffle=True)
for epoch in range(100):
    total_loss = 0
    for x, y in loader_rm:
        optimizer_rm.zero_grad()
        pred = reward_model(x)
        loss = criterion_rm(pred, y)
        loss.backward()
        optimizer_rm.step()
        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Reward model training epoch {epoch}, loss {total_loss/len(loader_rm):.4f}")
print("Reward model trained.")

# Function to get reward for a MIDI file using the trained reward model
def get_reward(midi_path):
    feats = extract_midi_features(midi_path)
    feats_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        reward = reward_model(feats_tensor).item()
    return reward


# ## 3. RLHF Training Loop (REINFORCE + KL penalty)

def compute_log_prob(sequence, genre_idx):
#log probability computation of a token list under current policy
# sequence: list of token ids
    tokens_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
    genre_tensor = torch.tensor([genre_idx], dtype=torch.long, device=device)
    logits = model.forward(tokens_tensor, genre_tensor)  # (1, T, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    log_lik = log_probs[:, :-1, :].gather(2, tokens_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_lik.sum().item()

def compute_kl_divergence(sequence, genre_idx):
#for the same sequence, KL between current policy and original policy
    log_prob_current = compute_log_prob(sequence, genre_idx)
    with torch.no_grad():
        # log probability under original model
        tokens_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
        genre_tensor = torch.tensor([genre_idx], dtype=torch.long, device=device)
        logits_orig = original_model.forward(tokens_tensor, genre_tensor)
        log_probs_orig = F.log_softmax(logits_orig, dim=-1)
        log_lik_orig = log_probs_orig[:, :-1, :].gather(2, tokens_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)
        log_prob_original = log_lik_orig.sum().item()
    return log_prob_current - log_prob_original   # KL approx

# RL hyperparameters
RL_LEARNING_RATE = 1e-5
rl_optimizer = torch.optim.Adam(model.parameters(), lr=RL_LEARNING_RATE)

print("Starting RLHF fine‑tuning...")
iteration_rewards = []
iteration_kls = []

for iter in range(NUM_RL_ITERATIONS):
    # Generate batch of samples
    batch_sequences = []
    batch_genres = []
    batch_rewards = []
    batch_tmp_paths = []
    
    model.eval()
    for _ in range(BATCH_SIZE_RL):
        idx = random.randint(0, len(val_token_ids)-1)
        seed_seq = val_token_ids[idx][:50].tolist()
        genre_idx = genre2idx[val_genres[idx]]
        gen_tokens = model.generate(seed_seq, genre_idx, max_new_tokens=GEN_LEN, top_p=TOP_P, temperature=TEMPERATURE)
        # Save temporarily to compute reward
        tmp_path = f"tmp_rl_{iter}_{_}.mid"
        tokens_to_midi(gen_tokens, tmp_path)
        reward = get_reward(tmp_path)
        batch_sequences.append(gen_tokens)
        batch_genres.append(genre_idx)
        batch_rewards.append(reward)
        batch_tmp_paths.append(tmp_path)
    
    # Normalize rewards
    rewards = np.array(batch_rewards)
    rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Compute policy gradient loss->policy gradient (sum of reward * grad(log prob)) + KL penalty
    model.train()
    total_loss = 0.0
    for seq, genre, rew_norm in zip(batch_sequences, batch_genres, rewards_normalized):
        # Compute log probability of the generated sequence
        tokens_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        genre_tensor = torch.tensor([genre], dtype=torch.long, device=device)
        logits = model.forward(tokens_tensor, genre_tensor)
        log_probs = F.log_softmax(logits, dim=-1)
        log_lik = log_probs[:, :-1, :].gather(2, tokens_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)
        log_prob_total = log_lik.sum()
      
        # KL penalty
        with torch.no_grad():
            logits_orig = original_model.forward(tokens_tensor, genre_tensor)
            log_probs_orig = F.log_softmax(logits_orig, dim=-1)
            log_lik_orig = log_probs_orig[:, :-1, :].gather(2, tokens_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)
            log_prob_orig_total = log_lik_orig.sum()
            kl_penalty = (log_prob_total - log_prob_orig_total) * KL_WEIGHT
        loss = -rew_norm * log_prob_total + kl_penalty
        total_loss += loss
        loss.backward()
    rl_optimizer.step()
    rl_optimizer.zero_grad()
    
    # Cleanup temporary files
    for p in batch_tmp_paths:
        if os.path.exists(p):
            os.remove(p)
    
    avg_reward = rewards.mean()
    avg_kl = (np.mean([compute_kl_divergence(s, g) for s,g in zip(batch_sequences, batch_genres)]))
    iteration_rewards.append(avg_reward)
    iteration_kls.append(avg_kl)
    print(f"Iteration {iter+1}/{NUM_RL_ITERATIONS} | Avg Reward: {avg_reward:.3f} | Avg KL: {avg_kl:.3f}")

torch.save(model.state_dict(), "models/transformer_rlhf.pt") # Save fine‑tuned model
print("RLHF model saved.")
plt.figure()
plt.plot(range(1, NUM_RL_ITERATIONS+1), iteration_rewards, marker='o')
plt.xlabel("RL Iteration")
plt.ylabel("Average Reward")
plt.title("Reward Progress during RLHF")
plt.grid()
plt.savefig("plots/rlhf_reward_progress.png")
plt.show()




# 4. Generate 10 Fine‑Tuned MIDI Files
model.eval()
print("Generating 10 fine‑tuned compositions...")
for i in range(10):
    idx = random.randint(0, len(val_token_ids)-1)
    seed_seq = val_token_ids[idx][:50].tolist()
    genre_idx = genre2idx[val_genres[idx]]
    gen_tokens = model.generate(seed_seq, genre_idx, max_new_tokens=GEN_LEN, top_p=TOP_P, temperature=TEMPERATURE)
    out_path = f"generated_midi_task4/rlhf_sample_{i+1}.mid"
    tokens_to_midi(gen_tokens, out_path)
    print(f"Saved {out_path}")




#5. Baseline (Task 3) vs RLHF Comparison

# Generating 10 samples from the original Task 3 model (without RL)- We reuse the Task 3 model (before RL) to generate another 10 files for comparison.
orig_model = original_model
orig_model.eval()
print("Generating 10 baseline (Task 3) compositions...")
for i in range(10):
    idx = random.randint(0, len(val_token_ids)-1)
    seed_seq = val_token_ids[idx][:50].tolist()
    genre_idx = genre2idx[val_genres[idx]]
    gen_tokens = orig_model.generate(seed_seq, genre_idx, max_new_tokens=GEN_LEN, top_p=TOP_P, temperature=TEMPERATURE)
    out_path = f"generated_midi_task4/baseline_task3_sample_{i+1}.mid"
    tokens_to_midi(gen_tokens, out_path)
    print(f"Saved baseline {out_path}")

# Compute metrics for both sets
def compute_metrics_for_dir(file_pattern):
    files = glob.glob(file_pattern)
    rhythms = []
    for f in files:
        try:
            midi = pretty_midi.PrettyMIDI(f)
            durations = []
            for inst in midi.instruments:
                for note in inst.notes:
                    durations.append(round(note.end - note.start, 2))
            if durations:
                rhythm_div = len(set(durations)) / len(durations)
                rhythms.append(rhythm_div)
        except:
            continue
    return np.mean(rhythms) if rhythms else 0.0

baseline_rhythm = compute_metrics_for_dir("generated_midi_task4/baseline_task3_sample_*.mid")
rlhf_rhythm = compute_metrics_for_dir("generated_midi_task4/rlhf_sample_*.mid")
print(f"Baseline (Task 3) rhythm diversity: {baseline_rhythm:.3f}")
print(f"RLHF‑tuned rhythm diversity: {rlhf_rhythm:.3f}")

task3_perplexity = 9.80   # obtained from Task 3 training

# For RLHF, we compute validation perplexity on the same validation set
model.eval()
total_val_loss = 0.0
# Need validation loader – simple approximate using original val_loader (we must recreate it from saved data)
# Since loading val_loader requires the dataset class, we'll reuse a simplified version:
class SimpleValDataset(Dataset):
    def __init__(self, token_ids, genres, max_len=MAX_SEQ_LEN):
        self.token_ids = token_ids
        self.genres = genres
        self.max_len = max_len
    def __len__(self):
        return len(self.token_ids)
    def __getitem__(self, idx):
        seq = self.token_ids[idx]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        genre = genre2idx[self.genres[idx]]
        return seq, genre
def collate_val(batch):
    seqs, genres = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
        mask[i, :len(s)] = True
    return padded, mask, torch.tensor(genres)
val_dataset = SimpleValDataset(val_token_ids, val_genres, MAX_SEQ_LEN)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_val)
total_val_loss = 0.0
with torch.no_grad():
    for tokens, mask, genres in val_loader:
        tokens, mask, genres = tokens.to(device), mask.to(device), genres.to(device)
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        target_mask = mask[:, 1:]
        logits = model(input_tokens, genres)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_tokens.reshape(-1), reduction='none')
        loss = loss.view(target_tokens.shape)
        loss = loss * target_mask.float()
        total_val_loss += loss.sum().item() / target_mask.float().sum().item()
val_loss = total_val_loss / len(val_loader)
rlhf_perplexity = np.exp(val_loss)
print(f"RLHF model validation perplexity: {rlhf_perplexity:.2f}")



# 6. Comparison Table 

#the average of scores from 10 participants manually inserted from the survey results
human_scores = {
    "Random Generator": 1.2,
    "Markov Chain": 2.7,
    "Task 1: Autoencoder": 3.2,
    "Task 2: VAE": 3.7,
    "Task 3: Transformer": 4.4,
    "Task 4: RLHF-Tuned": 4.7
}

# the comparison table with the acquired values from the tasks are manually inserted. Loss from t1,t2 and perplexity from t3, Rhythm Diversity
comparison = pd.DataFrame({
    "Model": list(human_scores.keys()),
    "Loss": ["—", "—", "0.0186", "0.0354", "—", "—"],
    "Perplexity": ["—", "—", "—", "—", "9.78", f"{rlhf_perplexity:.2f}"],
    "Rhythm Diversity": ["0", "0.330", "0.350", "0.333", "0.018", f"{rlhf_rhythm:.3f}"],
    "Human Score (1-5)": [f"{human_scores[m]:.1f}" for m in list(human_scores.keys())],
    "Genre Control": ["None", "Weak", "Single Genre", "Moderate", "Strong", "Strongest"]
})

comparison.to_csv("plots/task4_comparison_table.csv", index=False)
