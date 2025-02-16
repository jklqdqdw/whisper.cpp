#!/bin/bash

set -x
start_time=$(date +%s) # 开始时间

# Get the input file path and language from passed arguments
input_path=$1
language=${2:-english}

# Define the base directory for input files
base_input_dir="input"

# Extract filename from the input path and define new file paths
filename=$(basename "$input_path")

input_file="$base_input_dir/$filename"

# Copy the file to ./input directory, overwrite if it exists
cp -f "$input_path" "$input_file"

# Set the name of the output file for ffmpeg conversion
output_filename="converted_${filename%.*}.wav"
processed_input_file="$base_input_dir/$output_filename"

# Execute ffmpeg command
time ffmpeg -i "$input_file" -ar 16000 -ac 1 -c:a pcm_s16le "$processed_input_file"

# Define the name for the output text file
output_textfile_name="transcription_${filename%.*}.txt"
output_textfile_path="./output/$output_textfile_name"

# Ensure the ./output directory exists
mkdir -p "./output"

# Execute the main command with the processed input file and language parameter
time ./main -m models/ggml-large-v3.bin -f "$processed_input_file" --prompt "This document chronicles the proceedings of the meeting held by Michael's technical team; kindly disregard any ambient noise." --language "$language" > "$output_textfile_path"

# Remove timeline from the output text file
time sed -i '' 's/\[[0-9:\.]* --> [0-9:\.]*\]//g' "$output_textfile_path"

original_dir=$(dirname "$input_path")
cp "$output_textfile_path" "$original_dir/$output_textfile_name"

end_time=$(date +%s) # 结束时间
duration=$((end_time - start_time)) # 计算总用时
echo "Total Execution Time: $duration seconds"