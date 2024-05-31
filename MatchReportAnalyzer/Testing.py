import pytest
import matplotlib.pyplot as plt
from WebScraper import extract_text_from_url
from requests.exceptions import ConnectionError
from Tokenization import tokenize_text
from PoSTagging import pos_tag_text
from StopWordRemoval import remove_stopwords
from SentimentAnalyzer import perform_sentiment_analysis

# Test text acquisition
def test_valid_url():
    url = "https://www.theguardian.com/football/2024/apr/07/tottenham-nottingham-forest-premier-league-match-report"
    text = extract_text_from_url(url)
    if text:
        print("\nText Acquired")
    assert text, "Text extraction should be successful for a valid URL"

def test_invalid_url():
    invalid_url = "https://www.invalidurl.com"
    with pytest.raises(ConnectionError):
        extract_text_from_url(invalid_url)

# Test tokenization
def test_tokenization():
    text = "This is a sample sentence for tokenization."
    tokens = tokenize_text(text)
    expected_tokens = ['This', 'is', 'a', 'sample', 'sentence', 'for', 'tokenization', '.']
    print("\nExpected tokens:", expected_tokens)
    print("\nActual tokens:", tokens)
    assert tokens == expected_tokens, "Tokenization failed"

# Test part-of-speech tagging
def test_pos_tagging():
    tokens = ['This', 'is', 'a', 'sample', 'sentence', 'for', 'tokenization', '.']
    pos_tags = pos_tag_text(tokens)
    expected_tags = [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sample', 'JJ'), ('sentence', 'NN'), ('for', 'IN'), ('tokenization', 'NN'), ('.', '.')]
    print("\nExpected POS tags:", expected_tags)
    print("\nActual POS tags:", pos_tags)
    assert pos_tags == expected_tags, "Part-of-speech tagging failed"

# Test stop-word removal
def test_stopword_removal():
    tokens = ['This', 'is', 'a', 'sample', 'sentence', 'for', 'tokenization', '.']
    filtered_tokens = remove_stopwords(tokens)
    expected_result = ['sample', 'sentence', 'tokenization', '.']
    print("\nExpected filtered tokens:", expected_result)
    print("\nActual filtered tokens:", filtered_tokens)
    assert filtered_tokens == expected_result, "Stop-word removal failed"

# Test invalid sentiment analysis
def test_invalid_sentiment_analysis():
    # Empty text for sentiment analysis
    empty_text = []

    # Perform sentiment analysis on empty text
    sentiments, common_proper_nouns = perform_sentiment_analysis(empty_text)

    # Print actual sentiment scores
    print("\nSentiment scores:", sentiments)

    # Assert that sentiment analysis returns empty results
    assert not sentiments and not common_proper_nouns, "Invalid sentiment analysis failed"

# Test producing result/analysis (valid case)
def test_valid_graph_generation():
    # URL of the football match report
    url = "https://www.theguardian.com/football/2024/apr/07/tottenham-nottingham-forest-premier-league-match-report"

    # Extract text from the URL
    reportTextArr = extract_text_from_url(url)

    # Perform sentiment analysis
    sentiments, common_proper_nouns = perform_sentiment_analysis(reportTextArr)

    # Plot sentiment against count for proper nouns mentioned more than 2 times
    nouns = list(sentiments.keys())
    sentiment_scores = [sentiments[noun] for noun in nouns]
    noun_counts = [common_proper_nouns[noun] for noun in nouns]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(nouns)), sentiment_scores, color='orange')
    plt.xlabel('Proper Nouns Count')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment vs Count of Proper Nouns (Mentioned >= 2 times)')
    plt.xticks(range(len(nouns)), noun_counts)
    plt.grid(True)
    plt.tight_layout()

    # Check if the graph was successfully generated
    try:
        plt.show()
        print("\nGraph successfully generated")
    except Exception as e:
        pytest.fail("Graph generation failed: " + str(e))

# Test producing result/analysis (invalid case)
def test_invalid_graph_generation():
    # Attempt to generate a graph with empty data
    plt.bar([], [])
    plt.show()
    if not plt.gca().patches:
        print("\nEmpty graph generated successfully. Test passed.")
    else:
        print("\nEmpty graph generation failed. Test failed.")
        assert False, "Invalid graph generation succeeded"

if __name__ == '__main__':
    pytest.main()
