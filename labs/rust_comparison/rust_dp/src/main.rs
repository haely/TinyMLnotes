use csv::{ReaderBuilder, Writer};
use regex::Regex;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::time::Instant;
use lazy_static::lazy_static;

lazy_static! {
    static ref PUNC_REGEX : Regex  = Regex::new(r"[[:punct:]]").unwrap();
}

fn preprocess_text(text: &str, stop_words: &HashSet<String>, debug: bool) -> String {
    if debug {
        println!("🔹 Original Second Line: {}", text);
    }

    let no_punctuation = PUNC_REGEX.replace_all(text, "").to_string();
    if debug {
        println!("🔹 After Removing Punctuation: {}", no_punctuation);
    }

    let lowercase = no_punctuation.to_lowercase();
    if debug {
        println!("🔹 After Lowercasing: {}", lowercase);
    }

    let processed = lowercase
        .split_whitespace()
        .filter(|token| !stop_words.contains(*token))
        .fold("".to_string(), |acc, e| { 
            let mut n = acc.clone();
            n.push_str(" ");
            n.push_str(e);
            n
        });

    if debug {
        println!("🔹 After Removing Stopwords: {}", processed);
        println!("✅ Final Processed Line: {}\n", processed);
    }

    processed
}

fn process_csv(
    input_file: &str,
    output_file: &str,
    stop_words: &HashSet<String>,
    debug: bool,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    // Open input CSV
    let input = File::open(input_file)?;
    let mut rdr = ReaderBuilder::new().from_reader(BufReader::new(input));

    // Create output CSV
    let output = File::create(output_file)?;
    let mut wtr = Writer::from_writer(BufWriter::new(output));

    rdr.records().enumerate().try_for_each(|r| {
        wtr.write_record(r.1?.iter().map(
            |i| preprocess_text(i, stop_words, debug && 0 == r.0), // Debug second line
        ))
    })?;

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
