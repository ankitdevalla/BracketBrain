import os
import pandas as pd
import glob

# Define the headers for the CSV files
HEADERS = [
    "Team", "Adj OE", "Adj DE", "Barthag", "Record", "Wins", "Games", 
    "eFG", "eFG D.", "FT Rate", "FT Rate D", "TOV%", "TOV% D", "O Reb%", "Op OReb%", 
    "Raw T", "2P %", "2P % D.", "3P %", "3P % D.", "Blk %", "Blked %", "Ast %", 
    "Op Ast %", "3P Rate", "3P Rate D", "Adj. T", "Avg Hgt.", "Eff. Hgt.", "Exp.", 
    "Year", "PAKE", "PASE", "Talent", "", "FT%", "Op. FT%", "PPP Off.", "PPP Def.", 
    "Elite SOS"
]

def reset_and_add_headers(file_path):
    """
    Remove any existing headers and add the correct headers to a CSV file
    """
    try:
        # First, read the file to check if it has data
        try:
            # Try reading with no header first
            df = pd.read_csv(file_path, header=None)
        except:
            # If that fails, try with header
            df = pd.read_csv(file_path)
            # Convert the header row back to data if it was mistakenly read as header
            if len(df) > 0:
                headers = df.columns.tolist()
                df = pd.DataFrame([headers] + df.values.tolist())
        
        if len(df) == 0:
            print(f"Warning: File {file_path} appears to be empty. Skipping.")
            return
        
        # Determine if the first row looks like a header
        first_row_is_header = False
        if "Team" in str(df.iloc[0, 0]) or "Adj OE" in str(df.iloc[0, 1]):
            first_row_is_header = True
            # Remove the header row
            df = df.iloc[1:].reset_index(drop=True)
            print(f"Removed existing header from {file_path}")
        
        # Apply the correct headers
        if len(df.columns) <= len(HEADERS):
            df.columns = HEADERS[:len(df.columns)]
        else:
            print(f"Warning: More columns than expected in {file_path}: {len(df.columns)}")
            df = df.iloc[:, :len(HEADERS)]
            df.columns = HEADERS
        
        # Save the file with the correct headers
        df.to_csv(file_path, index=False)
        print(f"Applied correct headers to {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Path to T-Rank folder
    t_rank_folder = "../T-rank"
    
    if not os.path.exists(t_rank_folder):
        print(f"Error: T-Rank folder not found at {t_rank_folder}")
        return
    
    # Get all CSV files in the T-Rank folder
    csv_files = glob.glob(os.path.join(t_rank_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {t_rank_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {t_rank_folder}")
    
    # Process each file
    for file_path in csv_files:
        print(f"\nProcessing {os.path.basename(file_path)}...")
        reset_and_add_headers(file_path)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 