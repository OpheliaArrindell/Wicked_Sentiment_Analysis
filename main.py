import nltk
import json
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import pyplot as plt
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



# Ensure required NLTK resources are downloaded
nltk.download('all')

# Open the file with explicit UTF-8 encoding
with open('v__maguire__gregory_-_wicked.txt', 'r', encoding='utf-8') as text:
    wicked = text.read()

# Tokenize the text into words
words = word_tokenize(wicked)

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Filter out stopwords
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]

# Use Frequency Distribution function from NLTK on filtered words
freq_distribution = FreqDist(filtered_words)

# Part-of-speech tagging to identify proper nouns
tagged_text = pos_tag(filtered_words)  # Use filtered words for tagging

# Extract proper nouns (NNP = Proper Noun, Singular)
proper_nouns = [word for word, pos in tagged_text if pos == 'NNP']

# Filter and rank proper nouns by frequency
filtered_proper_nouns = {
    word: freq_distribution[word]
    for word in proper_nouns
}

# Remove anomalies (optional)
filtered_d = {
    word: freq
    for word, freq in filtered_proper_nouns.items()
    if freq > 0  # Exclude words with frequency 0
}

# Print the ranking of proper nouns
print("\nProper Nouns and Their Frequencies:")
print("-------------------------------------------")
for word, frequency in sorted(filtered_d.items(), key=lambda item: item[1], reverse=True):
    print(f"{word}: {frequency}")

# Create and rank a list of verbs in the book

ranked_list = [word for word, _ in freq_distribution.most_common()]

verbs = [word for word, pos in tagged_text if pos == 'VB']
# Eliminate the words that are also classified other than verbs
not_verbs = [word for word, pos in tagged_text if pos !='VB']

for v in verbs:
    if v in not_verbs:
        verbs.remove(v)

verb_d = {}

print('\n')
print("Verbs and their FD")
print("----------------------------------")

for word in ranked_list:
    if word in verbs:
        verb_d[word] = freq_distribution[word]
        print(word, ':', freq_distribution[word])


# putting the dictionary into a Dat Frame and printing out a bar graph

# Assuming `filtered_d` is your dictionary of data
df = pd.DataFrame.from_dict(filtered_d, orient='index', columns=['Frequency']).iloc[0:30]

# Print the DataFrame for verification
print(df)


# Plot the bar graph
df.plot.bar(figsize=(20, 10), legend=False, title='WICKED PROTAGONISTS')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()  # Display the bar graph

# Plot the pie chart
df.plot.pie(
    y='Frequency',
    figsize=(20, 10),
    legend=False,
    title='WICKED PROTAGONISTS',
    autopct='%1.1f%%'
)
plt.ylabel('')  # Remove the y-label for the pie chart
plt.tight_layout()
plt.show()  # Display the pie chart