import pandas as pd

# Read the raw data from the text file
input_file = "coaches2003.txt"
output_file = "coaches2003.csv"

# Read the data into a list of lines
with open(input_file, "r") as file:
    lines = file.readlines()

# Process the data and save as CSV
with open(output_file, "w") as file:
    for line in lines:
        # Replace multiple commas with a single comma
        cleaned_line = ",".join([item.strip() for item in line.split(",")])
        file.write(cleaned_line + "\n")