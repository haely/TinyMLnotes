# Comparison result
## Step 1

Wow!

```
❯ python3 main.py
🔹 Original Second Line: Tim McGraw
🔹 After Removing Punctuation: Tim McGraw
🔹 After Lowercasing: tim mcgraw
🔹 After Removing Stopwords: tim mcgraw
✅ Final Processed Line: tim mcgraw

🔹 Original Second Line: Put those Georgia stars to shame that night
🔹 After Removing Punctuation: Put those Georgia stars to shame that night
🔹 After Lowercasing: put those georgia stars to shame that night
🔹 After Removing Stopwords: put georgia stars shame night
✅ Final Processed Line: put georgia stars shame night

✅ Processing complete in 0.03 seconds
✅ Processed CSV saved to output.csv
```

```
❯ cargo run --release               
    Finished `release` profile [optimized] target(s) in 0.01s
     Running `target/release/rust_dp`
🔹 Original Second Line: Tim McGraw
🔹 After Removing Punctuation: Tim McGraw
🔹 After Lowercasing: tim mcgraw
🔹 After Removing Stopwords: tim mcgraw
✅ Final Processed Line: tim mcgraw

🔹 Original Second Line: Put those Georgia stars to shame that night
🔹 After Removing Punctuation: Put those Georgia stars to shame that night
🔹 After Lowercasing: put those georgia stars to shame that night
🔹 After Removing Stopwords: put georgia stars shame night
✅ Final Processed Line: put georgia stars shame night

✅ Processing complete in 0.20 seconds
✅ Processed CSV saved to output.csv
```

```
❯ cargo run          
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.01s
     Running `target/debug/rust_dp`
🔹 Original Second Line: Tim McGraw
🔹 After Removing Punctuation: Tim McGraw
🔹 After Lowercasing: tim mcgraw
🔹 After Removing Stopwords: tim mcgraw
✅ Final Processed Line: tim mcgraw

🔹 Original Second Line: Put those Georgia stars to shame that night
🔹 After Removing Punctuation: Put those Georgia stars to shame that night
🔹 After Lowercasing: put those georgia stars to shame that night
🔹 After Removing Stopwords: put georgia stars shame night
✅ Final Processed Line: put georgia stars shame night

✅ Processing complete in 1.46 seconds
✅ Processed CSV saved to output.csv
```
