## Unsupervised Neural Network for Multi-Genre Music Generation
This repository presents an end-to-end deep learning framework for symbolic music generation using unsupervised neural networks. The system generates realistic and musically coherent MIDI compositions across multiple genres without requiring manual labels.
The project explores multiple deep learning approaches, starting from basic sequence models to advanced transformer-based architectures and human feedback refinement.

## Project Idea
Music is a complex temporal structure involving melody, rhythm, harmony, and style. Instead of relying on labeled datasets, this project uses *Unsupervised learning* to discover hidden musical patterns directly from data.
The system learns the probability distribution of music sequences and generates new compositions that are both structured and expressive.

## Models Implemented
We explored four progressively advanced models:
- LSTM Autoencoder  
- Variational Autoencoder (VAE)  
- Transformer-based Music Generator  
- Reinforcement Learning from Human Feedback (RLHF)
- 
## Dataset
We used publicly available MIDI datasets:
- MAESTRO Dataset  
- Lakh MIDI Dataset  
Music data was processed using:
- Piano-roll representation  
- Tokenized event-based sequences  

## Tasks Overview

### Dataset & Preprocessing
- Collected and organized MIDI datasets  
- Cleaned and standardized music files  
- Converted MIDI into:
  - Piano-roll format  
  - Token-based sequence format  
- Prepared data for training different models  

### Task 1: LSTM Autoencoder
- Built sequence-to-sequence LSTM model  
- Learned compressed musical representations  
- Reconstructed and generated basic melodies  
### Task 2: Variational Autoencoder (VAE)
- Introduced probabilistic latent space  
- Improved diversity in generated music  
- Generated smoother and more varied compositions  
### Task 3: Transformer
- Implemented attention-based architecture  
- Captured long-term musical dependencies  
- Produced more structured and coherent multi-genre music  
### Task 4: RLHF
- Incorporated human feedback into training  
- Fine-tuned model based on listener preferences  
- Improved musical quality and human satisfaction  

### Baseline Models
- Simple statistical and rule-based sequence generators  
- Used for comparison with deep learning models  
- Helped evaluate improvements across architectures  

### Evaluation Metrics
We evaluated generated music using:
- Objective metrics (statistical structure, note distribution, repetition patterns)  
- Subjective evaluation (human listening feedback)  
- Long-term coherence and musical flow analysis  
- Listener preference scores (for RLHF stage)  

## Outcome
- LSTM captured local patterns  
- VAE improved diversity  
- Transformer achieved best long-term structure  
- RLHF further enhanced musical quality and human preference alignment  

## MIDI Files
Drive Link - https://drive.google.com/drive/folders/1PLj4sEb0cWGmNKNb-xzJX5BLAqOZDkAz?usp=drive_link


## Keywords
Music Generation, Deep Learning, LSTM, VAE, Transformer, RLHF, MIDI, Unsupervised Learning

## Future Improvements
- Add genre-controlled generation  
- Improve real-time music generation  
- Expand dataset diversity  
- Integrate audio synthesis from MIDI outputs  
