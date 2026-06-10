#!/usr/bin/env python3
"""
France Election Polarization Analysis - Text Analysis Module
Demonstrates text mining, sentiment analysis, topic modeling, and narrative classification.
Author: Sidra Jabeen Khan
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Text analysis libraries
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# For French text analysis (using English stopwords as proxy for demonstration)
# In actual research, would use spacy French model: spacy.load('fr_core_news_sm')
STOPWORDS = set(stopwords.words('english') + [
    'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'donc', 'pour',
    'dans', 'sur', 'avec', 'sans', 'par', 'ce', 'cette', 'ces', 'mon', 'ton',
    'son', 'notre', 'votre', 'leur', 'est', 'sont', 'etre', 'avoir', 'faire',
    'plus', 'moins', 'tres', 'trop', 'peu', 'beaucoup', 'tout', 'tous', 'toute',
    'toutes', 'autre', 'autres', 'meme', 'memes', 'quelque', 'quelques', 'tant',
    'tel', 'telle', 'tels', 'telles', 'chaque', 'chacun', 'chacune', 'plusieurs',
    'certains', 'certaines', 'aucun', 'aucune', 'nul', 'nulle', 'tout', 'tous',
    'toute', 'toutes', 'quel', 'quelle', 'quels', 'quelles', 'qui', 'que', 'quoi',
    'dont', 'ou', 'quand', 'comment', 'pourquoi', 'combien', 'lequel', 'laquelle',
    'lesquels', 'lesquelles', 'duquel', 'desquels', 'desquelles', 'auquel', 'auxquels',
    'auxquelles', 'a', 'de', 'en', 'vers', 'sous', 'devant', 'derriere', 'entre',
    'parmi', 'chez', 'contre', 'durant', 'pendant', 'apres', 'avant', 'depuis',
    'jusque', 'jusqu', 'malgre', 'selon', 'suivant', 'touchant', 'concernant',
    'excepte', 'outre', 'pres', 'voici', 'voila', 'y', 'en', 'lui', 'leur', 'eux',
    'elle', 'elles', 'nous', 'vous', 'je', 'tu', 'il', 'ils', 'on', 'me', 'te',
    'se', 'moi', 'toi', 'soi', 'lui', 'leur', 'eux', 'elle', 'elles', 'nous', 'vous'
])

def preprocess_text(text):
    """
    Clean and normalize social media text for analysis.
    Handles French text with English preprocessing as proxy.
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags (keep hashtag text for analysis)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    # Remove special characters, keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_keywords(text, min_length=3):
    """
    Extract keywords from preprocessed text.
    """
    words = text.split()
    keywords = [w for w in words if len(w) >= min_length and w not in STOPWORDS]
    return keywords

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER lexicon.
    Returns compound score: -1 (negative) to +1 (positive)
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    return scores['compound']

def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob.
    Returns polarity: -1 (negative) to +1 (positive)
    """
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

def classify_narrative(text, camp, polarization_score):
    """
    Classify narrative type based on content, political camp, and polarization.
    
    Categories:
    - dialogue_oriented: Constructive, bridge-building language
    - polarizing: Partisan but within democratic norms
    - radicalizing: Exclusionary, existential framing, anti-democratic undertones
    - mobilizing: Call to action, protest, electoral participation
    """
    text = str(text).lower()
    
    # Dialogue indicators
    dialogue_words = ['ensemble', 'unie', 'forte', 'progres', 'dialogue', 'debat', 
                      'democratie', 'citoyen', 'avenir', 'construire', 'ensemble',
                      'together', 'united', 'progress', 'dialogue', 'debate', 
                      'democracy', 'citizen', 'future', 'build']
    
    # Polarizing indicators  
    polarizing_words = ['contre', 'bataille', 'combat', 'lutte', 'ennemi', 'traite',
                        'menace', 'danger', 'proteger', 'frontiere', 'against', 
                        'battle', 'fight', 'struggle', 'enemy', 'traitor', 'threat',
                        'danger', 'protect', 'border']
    
    # Radicalizing indicators
    radicalizing_words = ['invasion', 'remigration', 'grand remplacement', 'trahison',
                          'soumission', 'occupation', 'resistance', 'revolution',
                          'invasion', 'remigration', 'great replacement', 'treason',
                          'submission', 'occupation', 'resistance', 'revolution']
    
    # Mobilizing indicators
    mobilizing_words = ['votez', 'mobilisez', 'manifestation', 'rassemblement',
                        'boycott', 'greve', 'action', 'vote', 'mobilize', 
                        'protest', 'gathering', 'boycott', 'strike', 'action']
    
    # Count indicators
    dialogue_count = sum(1 for w in dialogue_words if w in text)
    polarizing_count = sum(1 for w in polarizing_words if w in text)
    radicalizing_count = sum(1 for w in radicalizing_words if w in text)
    mobilizing_count = sum(1 for w in mobilizing_words if w in text)
    
    # Classification logic
    if radicalizing_count > 0 or polarization_score > 0.7:
        return 'radicalizing'
    elif polarizing_count > dialogue_count and polarization_score > 0.5:
        return 'polarizing'
    elif mobilizing_count > 0:
        return 'mobilizing'
    elif dialogue_count > 0 or polarization_score < 0.3:
        return 'dialogue_oriented'
    else:
        return 'polarizing'  # Default for moderate polarization

def perform_text_analysis(df):
    """
    Complete text analysis pipeline.
    """
    print("=== TEXT ANALYSIS PIPELINE ===")
    print(f"Analyzing {len(df)} posts...\n")
    
    # Preprocess text
    print("Step 1: Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    df['keywords'] = df['clean_text'].apply(extract_keywords)
    
    # Sentiment analysis
    print("Step 2: Sentiment analysis (VADER)...")
    df['sentiment_vader'] = df['clean_text'].apply(analyze_sentiment_vader)
    
    print("Step 3: Sentiment analysis (TextBlob)...")
    df['sentiment_textblob'] = df['clean_text'].apply(analyze_sentiment_textblob)
    
    # Average sentiment
    df['sentiment_avg'] = (df['sentiment_vader'] + df['sentiment_textblob']) / 2
    
    # Narrative classification
    print("Step 4: Narrative classification...")
    df['narrative_class'] = df.apply(
        lambda row: classify_narrative(row['clean_text'], row['camp'], row['polarization_score']),
        axis=1
    )
    
    # Word frequency analysis
    print("Step 5: Word frequency analysis...")
    all_words = []
    for keywords in df['keywords']:
        all_words.extend(keywords)
    word_freq = Counter(all_words)
    
    print(f"\nTop 20 most frequent words:")
    for word, count in word_freq.most_common(20):
        print(f"  {word}: {count}")
    
    return df, word_freq

def visualize_text_analysis(df, word_freq):
    """
    Create visualizations for text analysis results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sentiment distribution by political camp
    camp_sentiment = df.groupby('camp')['sentiment_avg'].mean().sort_values()
    camp_sentiment.plot(kind='barh', ax=axes[0,0], color='steelblue')
    axes[0,0].set_title('Average Sentiment by Political Camp')
    axes[0,0].set_xlabel('Sentiment Score (-1 to +1)')
    
    # 2. Narrative classification distribution
    narrative_counts = df['narrative_class'].value_counts()
    colors = {'dialogue_oriented': 'green', 'mobilizing': 'blue', 
              'polarizing': 'orange', 'radicalizing': 'red'}
    narrative_counts.plot(kind='bar', ax=axes[0,1], 
                          color=[colors.get(c, 'gray') for c in narrative_counts.index])
    axes[0,1].set_title('Narrative Classification Distribution')
    axes[0,1].set_xlabel('Narrative Type')
    axes[0,1].set_ylabel('Number of Posts')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Polarization vs Sentiment scatter
    for camp in df['camp'].unique():
        subset = df[df['camp'] == camp]
        axes[1,0].scatter(subset['polarization_score'], subset['sentiment_avg'], 
                         label=camp, alpha=0.6, s=30)
    axes[1,0].set_title('Polarization vs Sentiment by Camp')
    axes[1,0].set_xlabel('Polarization Score')
    axes[1,0].set_ylabel('Sentiment Score')
    axes[1,0].legend()
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1,0].axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label='High polarization')
    
    # 4. Top words bar chart
    top_words = [word for word, count in word_freq.most_common(15)]
    top_counts = [count for word, count in word_freq.most_common(15)]
    axes[1,1].barh(top_words, top_counts, color='coral')
    axes[1,1].set_title('Top 15 Most Frequent Words')
    axes[1,1].set_xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('text_analysis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'text_analysis_results.png'")

if __name__ == "__main__":
    # Load data
    print("Loading simulated data...")
    df = pd.read_csv('france_election_2022_simulated.csv')
    
    # Perform analysis
    df, word_freq = perform_text_analysis(df)
    
    # Visualize
    visualize_text_analysis(df, word_freq)
    
    # Summary statistics
    print("\n=== NARRATIVE CLASSIFICATION SUMMARY ===")
    print(df['narrative_class'].value_counts())
    print(f"\nPercentage of radicalizing content: {(df['narrative_class'] == 'radicalizing').mean()*100:.1f}%")
    print(f"Percentage of dialogue-oriented content: {(df['narrative_class'] == 'dialogue_oriented').mean()*100:.1f}%")
    
    # Save processed data
    df.to_csv('france_election_2022_processed.csv', index=False, encoding='utf-8')
    print("\nProcessed data saved to 'france_election_2022_processed.csv'")
