import json
import re
import codecs

# Path to your JSON file
input_file = "../pre_tourney_data/team_sum.json"
output_file = "../pre_tourney_data/team_sum_clean.json"

# Load the JSON file
with open(input_file, 'r', encoding='utf-8') as f:
    team_summaries = json.load(f)

# Function to clean the text
def clean_text(text):
    # Remove all contentReference patterns
    cleaned = re.sub(r'&#8203;:contentReference\[.*?\]\{.*?\}', '', text)
    cleaned = re.sub(r':contentReference\[.*?\]\{.*?\}', '', cleaned)
    
    # Replace common problematic characters
    cleaned = cleaned.replace('\u2013', '-')  # Replace en dash with hyphen
    cleaned = cleaned.replace('\u2014', '-')  # Replace em dash with hyphen
    cleaned = cleaned.replace('\u2018', "'")  # Replace left single quote
    cleaned = cleaned.replace('\u2019', "'")  # Replace right single quote
    cleaned = cleaned.replace('\u201c', '"')  # Replace left double quote
    cleaned = cleaned.replace('\u201d', '"')  # Replace right double quote
    
    return cleaned

# Clean each team summary
cleaned_summaries = {}
for team_id, summary in team_summaries.items():
    cleaned_summaries[team_id] = clean_text(summary)

# Save the cleaned JSON with ensure_ascii=False to preserve actual characters
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_summaries, f, indent=2, ensure_ascii=False)

print(f"Cleaned JSON saved to {output_file}")