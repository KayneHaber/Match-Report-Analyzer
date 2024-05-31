import WebScraper
import SentimentAnalyzer
import matplotlib.pyplot as plt

# URL of the football match report
url = "https://www.theguardian.com/football/article/2024/may/29/olympiakos-fiorentina-europa-conference-league-final-match-report"

# Extract text from the URL
reportTextArr = WebScraper.extract_text_from_url(url)

# Perform sentiment analysis
sentiments, common_proper_nouns = SentimentAnalyzer.perform_sentiment_analysis(reportTextArr)

# Print the text of the report
print("Match Report:\n", "\n".join(reportTextArr))

# Print a couple of new lines
print("\n" * 2)

# Print the array of proper nouns and their sentiment scores
print("Nouns and sentiment:")
for noun, sentiment_score in sentiments.items():
    print(noun, " sentiment:", sentiment_score)

# Plot
plt.figure(figsize=(10, 6))

# Plot sentiment against count for proper nouns mentioned more than 2 times
nouns = list(sentiments.keys())
sentiment_scores = [sentiments[noun] for noun in nouns]
noun_counts = [common_proper_nouns[noun] for noun in nouns]

plt.bar(range(len(nouns)), sentiment_scores, color='orange')
plt.xlabel('Proper Nouns Count')
plt.ylabel('Sentiment Score')
plt.title('Sentiment vs Count of Proper Nouns (Mentioned >= 2 times)')

# Annotate each bar with the corresponding noun
for i, (count, score) in enumerate(zip(noun_counts, sentiment_scores)):
    plt.text(i, score, nouns[i], ha='center', va='bottom')

plt.xticks(range(len(nouns)), noun_counts)
plt.grid(True)
plt.tight_layout()
plt.show()