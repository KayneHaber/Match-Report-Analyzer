import Tokenization
import PoSTagging
import StopWordRemoval
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Function to count proper nouns mentioned more than 2 times
def count_common_proper_nouns(tokens):
    proper_nouns = [word for word, pos in PoSTagging.pos_tag_text(tokens) if pos == 'NNP' and word.isalpha()]
    proper_noun_count = Counter(proper_nouns)
    common_proper_nouns = {noun: count for noun, count in proper_noun_count.items() if count >= 2}
    return common_proper_nouns

# Function to perform sentiment analysis
def perform_sentiment_analysis(reportTextArr):
    # Join paragraphs to form a single string
    full_text = "\n".join(reportTextArr)

    # Tokenization
    tokens = Tokenization.tokenize_text(full_text)

    # PoS Tagging
    pos_tags = PoSTagging.pos_tag_text(tokens)

    # Stop-word removal
    filtered_tokens = StopWordRemoval.remove_stopwords(tokens)

    # Find proper nouns mentioned more than 2 times
    common_proper_nouns = count_common_proper_nouns(filtered_tokens)

    # Sentiment analysis using VADER for each common proper noun
    analyzer = SentimentIntensityAnalyzer()
    sentiments = {}
    for noun in common_proper_nouns:
        noun_sentiments = []
        for sentence in reportTextArr:
            if noun in sentence:
                sentiment_score = analyzer.polarity_scores(sentence)['compound']
                noun_sentiments.append(sentiment_score)
        sentiments[noun] = sum(noun_sentiments) / len(noun_sentiments)

    return sentiments, common_proper_nouns
