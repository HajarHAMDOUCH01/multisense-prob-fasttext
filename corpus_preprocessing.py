import nltk
nltk.download('punkt')        # For tokenization
nltk.download('stopwords')    # For stopwords
nltk.download('wordnet')      # For lemmatization
nltk.download('omw-1.4')      # Open Multilingual Wordnet (for WordNetLemmatizer)
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    """
    Performs a series of cleaning operations on a given text string.
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    # Use str.translate to remove all punctuation quickly
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Remove numbers (optional, depending on your needs)
    text = re.sub(r'\d+', '', text)

    # 4. Tokenize the text
    tokens = word_tokenize(text)

    # 5. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 6. Lemmatization (reduce words to their base form)
    # Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 7. Remove empty strings or tokens that are just whitespace after cleaning
    tokens = [token for token in tokens if token.strip()]

    # 8. Join tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

def clean_corpus(input_filepath, output_filepath):
    """
    Reads text from an input file, cleans each line, and writes to an output file.
    """
    print(f"Starting cleaning process for: {input_filepath}")
    cleaned_lines_count = 0
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for line_num, line in enumerate(infile):
                    if line_num % 10000 == 0 and line_num > 0:
                        print(f"Processed {line_num} lines...")
                    cleaned_line = clean_text(line)
                    outfile.write(cleaned_line + '\n')
                    cleaned_lines_count += 1
        print(f"Cleaning complete. {cleaned_lines_count} lines processed and saved to: {output_filepath}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Replace 'data.txt' with the path to your input corpus file
    input_file = 'data.txt'
    # Replace 'cleaned_data.txt' with the desired path for the cleaned output file
    output_file = 'cleaned_data.txt'

    # --- Run the cleaning process ---
    clean_corpus(input_file, output_file)