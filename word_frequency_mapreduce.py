import string
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import requests
import matplotlib.pyplot as plt

# Step 1: Download text from a URL
def get_text(url):
    """
    Downloads text from a URL.
    
    :param url: URL to download the text from
    :return: Text content as a string, or None if an error occurs
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        return response.text
    except requests.RequestException as e:
        print(f"Error downloading text: {e}")
        return None

# Step 2: Remove punctuation
def remove_punctuation(text):
    """
    Removes punctuation from the text.

    :param text: Input text
    :return: Text without punctuation
    """
    return text.translate(str.maketrans("", "", string.punctuation))

# Step 3: MapReduce functions
def map_function(word):
    """
    Maps a word to a key-value pair (word, 1).

    :param word: Word to process
    :return: Tuple (word, 1)
    """
    return word, 1

def shuffle_function(mapped_values):
    """
    Groups values by key (word) for reduction.

    :param mapped_values: List of mapped key-value pairs
    :return: Iterable of grouped items (word, list of counts)
    """
    shuffled = defaultdict(list)
    for key, value in mapped_values:
        shuffled[key].append(value)
    return shuffled.items()

def reduce_function(key_values):
    """
    Reduces key-value pairs to a single key with the sum of values.

    :param key_values: Tuple (key, list of values)
    :return: Tuple (key, sum of values)
    """
    key, values = key_values
    return key, sum(values)

def map_reduce(text):
    """
    Executes the MapReduce process on the input text.

    :param text: Input text to process
    :return: Dictionary with word frequencies
    """
    # Remove punctuation and split text into words
    text = remove_punctuation(text)
    words = text.split()

    # Parallel Mapping
    with ThreadPoolExecutor() as executor:
        mapped_values = list(executor.map(map_function, words))

    # Shuffle step
    shuffled_values = shuffle_function(mapped_values)

    # Parallel Reduction
    with ThreadPoolExecutor() as executor:
        reduced_values = list(executor.map(reduce_function, shuffled_values))

    return dict(reduced_values)

# Step 4: Visualize the results
def visualize_top_words(word_counts, top_n=10):
    """
    Visualizes the top N most frequent words using a bar chart.

    :param word_counts: Dictionary with word frequencies
    :param top_n: Number of top words to visualize
    """
    # Get the top N words
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*sorted_words)

    # Plot the words and their counts
    plt.barh(words, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.gca().invert_yaxis()
    plt.show()

# Main script
if __name__ == "__main__":
    # URL of the text to analyze
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"  # Example: Pride and Prejudice

    # Step 1: Download the text
    print("Downloading text...")
    text = get_text(url)

    if text:
        # Step 2: Perform MapReduce
        print("Processing text with MapReduce...")
        word_counts = map_reduce(text)

        # Step 3: Visualize the top words
        print("Visualizing top words...")
        visualize_top_words(word_counts, top_n=10)
    else:
        print("Error: Failed to download the text.")
