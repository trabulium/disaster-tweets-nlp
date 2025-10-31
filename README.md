# NLP Disaster Tweets Classification
## Using Recurrent Neural Networks for Tweet Classification

**Author**: Matthew Campbell
**GitHub**: https://github.com/trabulium/disaster-tweets-nlp
**Competition**: [Kaggle - NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

---

## Project Overview

This project tackles the challenge of identifying whether tweets are about real disasters or not using Natural Language Processing and Recurrent Neural Networks (RNNs). The model analyses tweet text to classify them into disaster (1) or non-disaster (0) categories.

### Why This Matters
During emergency situations, social media becomes a critical communication channel. Automatically identifying genuine disaster tweets can help:
- Emergency services respond faster
- News organisations verify events quickly
- Aid organisations deploy resources more effectively

---

## Problem Statement

**Task**: Binary text classification
**Dataset**: 7,613 labelled tweets (training), 3,263 unlabelled (test)
**Evaluation Metric**: F1 Score
**Approaches**: LSTM, GRU, Bidirectional RNNs, GloVe embeddings

---

## Results Summary

### Final Performance

| Model | Validation F1 | Validation AUC | Kaggle F1 | Notes |
|-------|--------------|----------------|-----------|-------|
| Baseline LSTM | 0.78 | 0.83 | - | Severe plateau (14 epochs stuck) |
| GRU | 0.00 | 0.50 | - | Complete failure despite fixes |
| Bidirectional LSTM | 0.79 | 0.85 | 0.78394 | Stable training, no plateau |
| BiLSTM + Enhanced Preprocessing | - | - | 0.79344 | +0.0095 improvement |
| **BiLSTM + GloVe Embeddings** | **0.79** | **0.89** | **0.80937** | **Best model** |

**Final Kaggle Score: 0.80937 F1**
**Rank: 225/815 (Top 27.6%)**

---

## Model Architectures

### 1. Baseline LSTM
- Simple LSTM layer (64 units)
- Embedding layer (100-dim, trained from scratch)
- Dense output with sigmoid activation
- **Issue**: Severe plateau behavior - stuck at 50% accuracy for 14 epochs before breakthrough

### 2. GRU Model
- Gated Recurrent Unit (64 units)
- Simpler architecture than LSTM
- **Result**: Complete failure (42.94% accuracy, AUC 0.50)
- **Attempted fixes**: Gradient clipping, orthogonal initialization, ReLU activation, learning rate reduction - all failed

### 3. Bidirectional LSTM
- Processes text in both directions
- Better context understanding
- **Breakthrough**: No plateau - learned immediately from epoch 1
- More robust to poor initialization

### 4. BiLSTM + GloVe Embeddings (BEST)
- Pre-trained Twitter embeddings (27B tweets, 100-dim)
- 84.1% vocabulary coverage
- Frozen embeddings (trainable=False)
- **Result**: Most stable training, best performance

---

## Key Improvements

### What Worked

**1. Enhanced Text Preprocessing (+0.0095 F1)**
- Stopwords removal using NLTK
- Lemmatization (NOT stemming - stemming produced poor results)
- Punctuation removal
- URL and mention removal

**2. GloVe Pre-trained Embeddings (+0.0156 F1)** - Biggest win!
- Jumped from 0.79344 → 0.80937 F1
- Pre-trained on 2 billion tweets - perfect domain match
- Eliminated initialization issues
- Fastest, most stable training

**3. Bidirectional Architecture**
- Immediate learning (no plateau)
- Better gradient flow from dual LSTM paths
- Richer context understanding

### What Didn't Work

**1. GRU Architecture** - Complete failure
- Stuck at 42.94% accuracy (random guessing level)
- Tried multiple fixes: gradient clipping (0.5, 1.0), ReLU activation, orthogonal initialization, reduced learning rate
- Conclusion: GRU not suitable for this dataset/task

**2. Stemming** - Produced nonsensical words
- "earthquake" → "earthquak"
- "cities" → "citi"
- Lemmatization preserved actual words and performed better

**3. Hyperparameter Tuning** - Diminishing returns
- Larger models (150-dim embeddings, 96 units): Worse performance (0.78240 F1)
- Dataset too small (6,090 samples) to benefit from increased capacity
- Better data (GloVe) beat bigger models

---

## Training Insights

### RNN Training Instability

**Critical Discovery**: Same architecture, same hyperparameters → vastly different results due to random initialization:
- Run 1: Breakthrough at epoch 5, final 79% accuracy
- Run 2: Stuck for 14 epochs, then breakthrough, final 79% accuracy
- Run 3: Never escaped plateau, 42.9% accuracy (GRU)

**Solution**: Bidirectional LSTM + pre-trained embeddings eliminated this unpredictability

### Threshold Optimization

Tested thresholds 0.30-0.70 for classification cutoff:
- Best model (GloVe BiLSTM): Optimal at 0.50 (perfectly calibrated!)
- Poorer runs: Required 0.55-0.65 for best F1 (poor calibration)
- Improvement: +0.00% to +0.37% depending on model quality

---

## Key Learnings

### Technical Insights

1. **Pre-trained embeddings > Larger models** - GloVe improvement (+0.0156 F1) exceeded all hyperparameter tuning attempts
2. **Bidirectional > Unidirectional** - More stable, no plateau issues
3. **RNN training is highly sensitive to initialization** - Multiple runs necessary to assess true performance
4. **When debugging fails, move on** - Spent 2 hours debugging GRU, 30 minutes implementing GloVe had better ROI
5. **Small datasets need different strategies** - Can't learn good representations, must use pre-trained

### Practical Lessons

- Validation scores ≠ Kaggle scores (different test distributions)
- Random seed variance: ±0.003 F1 across runs
- Class imbalance (57/43) manageable with class weights
- Well-calibrated models have optimal threshold at 0.5

---

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Jupyter Notebook or Google Colab

### Installation

```bash
# Clone repository
git clone https://github.com/trabulium/disaster-tweets-nlp.git
cd disaster-tweets-nlp

# Download data from Kaggle
kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip -d data/

# Run notebook
jupyter notebook disaster_tweets_nlp.ipynb
```

### For GloVe Embeddings

Download GloVe Twitter embeddings (100-dimensional):
```bash
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
# Use glove.twitter.27B.100d.txt
```

---

## Future Improvements

1. **Transformer models**: Fine-tune BERT or RoBERTa (+0.05-0.08 F1 expected)
2. **Attention mechanisms**: Add attention layers to highlight important words
3. **Ensemble methods**: Combine BiLSTM + GloVe with other strong models
4. **Advanced preprocessing**: Spell correction, emoji handling, hashtag splitting
5. **Data augmentation**: Back-translation, synonym replacement to expand training set

---

## Resources

- [Kaggle Competition](https://www.kaggle.com/c/nlp-getting-started)
- [Understanding LSTM Networks (Colah's Blog)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [TensorFlow RNN Guide](https://www.tensorflow.org/guide/keras/rnn)

---

## Acknowledgements

- Kaggle for hosting the competition and providing the dataset
- Stanford NLP for GloVe pre-trained embeddings
- TensorFlow/Keras team for excellent documentation

---

**Last Updated**: October 2025
