import csv
import re
import time

def preprocess_text(text, stop_words, debug=False):
    """Preprocess text: remove punctuation, convert to lowercase, remove stopwords."""
    
    if debug:
        print(f"🔹 Original Second Line: {text}")

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', text)

    if debug:
        print(f"🔹 After Removing Punctuation: {no_punctuation}")

    # Convert to lowercase
    lowercase = no_punctuation.lower()

    if debug:
        print(f"🔹 After Lowercasing: {lowercase}")

    # Tokenize and remove stopwords
    tokens = [word for word in lowercase.split() if word not in stop_words]
    processed = " ".join(tokens)

    if debug:
        print(f"🔹 After Removing Stopwords: {processed}")
        print(f"✅ Final Processed Line: {processed}\n")

    return processed

def process_csv(input_file, output_file, stopwords_file, debug=False):
    """Reads input CSV, processes text, and writes to output CSV."""
    
    start_time = time.time()

    # Load stopwords into a set
    with open(stopwords_file, "r", encoding="utf-8") as f:
        stop_words = set(line.strip().lower() for line in f)

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8", newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            processed_row = [preprocess_text(col, stop_words, debug=(debug and i == 1)) for col in row]
            writer.writerow(processed_row)

    elapsed_time = time.time() - start_time
    print(f"✅ Processing complete in {elapsed_time:.2f} seconds")

# Example usage
if __name__ == "__main__":
    input_csv = "input.csv"
    output_csv = "output.csv"
    stopwords_txt = "stopwords.txt"

    process_csv(input_csv, output_csv, stopwords_txt, debug=True)

    print(f"✅ Processed CSV saved to {output_csv}")
