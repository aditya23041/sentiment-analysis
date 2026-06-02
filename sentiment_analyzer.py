"""
Sentiment Analysis Module using NLP
Analyzes text sentiment using NLTK and TextBlob
"""

import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from typing import Dict, List, Tuple

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SentimentAnalyzer:
    """Analyze sentiment of text using NLTK and TextBlob"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob polarity and subjectivity
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with polarity, subjectivity, and sentiment label
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = "POSITIVE"
        elif polarity < -0.1:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return {
            'text': text,
            'polarity': round(polarity, 4),
            'subjectivity': round(subjectivity, 4),
            'sentiment': sentiment
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by tokenizing and removing stopwords
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stopwords and punctuation
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return ' '.join(tokens)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze_textblob(text) for text in texts]
    
    def analyze_from_csv(self, filepath: str, text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze sentiment from CSV file
        
        Args:
            filepath: Path to CSV file
            text_column: Name of column containing text
            
        Returns:
            DataFrame with sentiment analysis results
        """
        df = pd.read_csv(filepath)
        results = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            result = self.analyze_textblob(text)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        return results_df


def main():
    """Example usage of SentimentAnalyzer"""
    
    analyzer = SentimentAnalyzer()
    
    # Example 1: Single text analysis
    print("=" * 60)
    print("SINGLE TEXT ANALYSIS")
    print("=" * 60)
    test_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. Complete waste of money.",
        "The weather is okay today.",
        "I absolutely hate waiting in long lines!",
        "Best purchase I've ever made!"
    ]
    
    for text in test_texts:
        result = analyzer.analyze_textblob(text)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Polarity: {result['polarity']} (Range: -1 to 1)")
        print(f"Subjectivity: {result['subjectivity']} (Range: 0 to 1)")
    
    # Example 2: Batch analysis
    print("\n" + "=" * 60)
    print("BATCH ANALYSIS")
    print("=" * 60)
    results = analyzer.analyze_batch(test_texts)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Example 3: Preprocessed text
    print("\n" + "=" * 60)
    print("TEXT PREPROCESSING")
    print("=" * 60)
    sample_text = "I absolutely love this amazing product! It's really wonderful."
    preprocessed = analyzer.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Preprocessed: {preprocessed}")


if __name__ == "__main__":
    main()
