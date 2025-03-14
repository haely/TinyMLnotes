use csv::{ReaderBuilder, Writer};
use regex::Regex;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::time::Instant;

fn preprocess_text(text: &str, stop_words: &HashSet<String>, debug: bool) -> String {
    let punctuation_regex = Regex::new(r"[[:punct:]]").unwrap();

    if debug {
        println!("🔹 Original Second Line: {}", text);
    }

    let no_punctuation = punctuation_regex.replace_all(text, "").to_string();
    if debug {
        println!("🔹 After Removing Punctuation: {}", no_punctuation);
    }

    let lowercase = no_punctuation.to_lowercase();
    if debug {
        println!("🔹 After Lowercasing: {}", lowercase);
    }

    let tokens: Vec<String> = lowercase
        .split_whitespace()
        .filter(|token| !stop_words.contains(*token))
        .map(|s| s.to_string())
        .collect();

    let processed = tokens.join(" ");
    if debug {
        println!("🔹 After Removing Stopwords: {}", processed);
        println!("✅ Final Processed Line: {}\n", processed);
    }

    processed
}

fn process_csv(input_file: &str, output_file: &str, stop_words: &HashSet<String>, debug: bool) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    // Open input CSV
    let input = File::open(input_file)?;
    let mut rdr = ReaderBuilder::new().from_reader(BufReader::new(input));

    // Create output CSV
    let output = File::create(output_file)?;
    let mut wtr = Writer::from_writer(BufWriter::new(output));

    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let mut processed_record = Vec::new();

        for (_j, field) in record.iter().enumerate() {
            let processed = preprocess_text(field, stop_words, debug && i == 1);  // Debug second line
            processed_record.push(processed);
        }

        // Write processed row to the output CSV
        wtr.write_record(&processed_record)?;
    }

    // Ensure all writes are flushed
    wtr.flush()?;

    let elapsed = start.elapsed().as_secs_f64();
    println!("✅ Processing complete in {:.2} seconds", elapsed);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let input_csv = "input.csv";
    let output_csv = "output.csv";
    
    // Load stopwords from a file
    let stopwords_file = "stopwords.txt";
    let stopwords_content = std::fs::read_to_string(stopwords_file)?;
    let stop_words: HashSet<String> = stopwords_content.lines().map(String::from).collect();

    // Process CSV
    process_csv(input_csv, output_csv, &stop_words, true)?;

    println!("✅ Processed CSV saved to {}", output_csv);
    Ok(())
}


