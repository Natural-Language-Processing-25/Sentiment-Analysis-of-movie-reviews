# NLP Project    
import pandas as pd
import os
from pathlib import Path
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def create_sentiment_dataset():
    texts = []
    labels = []
    word_tokens = []
    sentence_tokens = []
    
    base_path = Path('review_polarity/txt_sentoken')
    pos_path = base_path / 'pos'
    neg_path = base_path / 'neg'
    
    if pos_path.exists():
        for file in pos_path.glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)

                doc = nlp(text)
                word_tokens.append([token.text.lower() for token in doc if not token.is_punct and not token.is_space])
                sentence_tokens.append([sent.text.strip() for sent in doc.sents])
                labels.append(1)
    
    if neg_path.exists():
        for file in neg_path.glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)

                doc = nlp(text)
                word_tokens.append([token.text.lower() for token in doc if not token.is_punct and not token.is_space])
                sentence_tokens.append([sent.text.strip() for sent in doc.sents])
                labels.append(0)
    
    df = pd.DataFrame({
        'text': texts,
        'word_tokens': word_tokens,
        'sentence_tokens': sentence_tokens,
        'label': labels
    })
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Create the dataset
    df = create_sentiment_dataset()
    
    # Display first few rows and basic information
    print("\nFirst few rows of the dataset:")
    df.head()
    print("\nDataset shape:", df.shape)
    # print("\nLabel distribution:")
    # print(df['label'].value_counts())
    
    # # Display tokenization examples
    # print("\nExample of word tokenization for first review:")
    # print("Number of words:", len(df['word_tokens'][0]))
    # print("First 20 words:", df['word_tokens'][0][:20])
    
    # print("\nExample of sentence tokenization for first review:")
    # print("Number of sentences:", len(df['sentence_tokens'][0]))
    # print("First 2 sentences:", df['sentence_tokens'][0][:2])    
