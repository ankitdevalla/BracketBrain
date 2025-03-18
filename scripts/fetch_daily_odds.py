import requests
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import time
import argparse
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_team_mappings(teams_file='../raw_data/MTeams.csv', mapping_json='../pre_tourney_data/mapping.json'):
    """
    Load team mappings from MTeams.csv and mapping.json
    
    Parameters:
    - teams_file: Path to the MTeams.csv file
    - mapping_json: Path to the mapping.json file with direct name to ID mappings
    
    Returns:
    - Dictionary mapping team names to team IDs
    """
    team_mappings = {}
    
    # First try to load the direct mappings from the JSON file
    try:
        with open(mapping_json, 'r') as f:
            direct_mappings = json.load(f)
            print(f"Loaded {len(direct_mappings)} direct team mappings from {mapping_json}")
            
            # Add direct mappings to the team_mappings dictionary with highest priority
            for team_name, team_id in direct_mappings.items():
                team_mappings[team_name.lower()] = team_id
                
                # Also add the team name without the suffix for better matching
                # For example, "North Carolina Tar Heels" -> "north carolina"
                parts = team_name.lower().split()
                if len(parts) > 2:  # Only process if the name has multiple parts
                    potential_base_name = " ".join(parts[:-1])  # Drop the last part (likely suffix)
                    team_mappings[potential_base_name] = team_id
                
    except Exception as e:
        print(f"Warning: Could not load mapping JSON file: {str(e)}")
    
    # Then load the MTeams.csv file for additional mappings (lower priority)
    try:
        # Read the teams CSV file
        teams_df = pd.read_csv(teams_file)
        print(f"Loaded {len(teams_df)} teams from {teams_file}")
        
        # Process each team name, but only add if not already mapped
        for _, row in teams_df.iterrows():
            team_id = row['TeamID']
            team_name = row['TeamName']
            
            # Only add if not already in mappings
            if team_name.lower() not in team_mappings:
                team_mappings[team_name.lower()] = team_id
                
                # Add minimal common variations
                if 'St ' in team_name:
                    if team_name.lower().replace('st ', 'saint ') not in team_mappings:
                        team_mappings[team_name.lower().replace('st ', 'saint ')] = team_id
                    if team_name.lower().replace('st ', 'state ') not in team_mappings:
                        team_mappings[team_name.lower().replace('st ', 'state ')] = team_id
        
        print(f"Final mapping dictionary contains {len(team_mappings)} entries")
        return team_mappings
    except Exception as e:
        print(f"Error loading team mappings from CSV: {str(e)}")
        # If CSV loading fails, still return any mappings from the JSON
        return team_mappings

def match_team_name(team_name, team_mappings):
    """
    Match a team name from the odds data to a team ID
    
    Parameters:
    - team_name: Name of the team from the odds data
    - team_mappings: Dictionary mapping team names to team IDs
    
    Returns:
    - Team ID if found, None otherwise
    """
    if not team_name or not team_mappings:
        return None
        
    # Clean up the team name
    cleaned_name = team_name.lower().strip()
    
    # Direct match check - exact match for the full name (highest priority)
    if cleaned_name in team_mappings:
        return team_mappings[cleaned_name]
    
    # Try minor variations of the team name
    name_variations = [
        cleaned_name,
        cleaned_name.replace('st.', 'saint'),
        cleaned_name.replace('st.', 'st'),
        cleaned_name.replace('st.', 'state'),
        cleaned_name.replace('&', 'and'),
        cleaned_name.replace('and', '&')
    ]
    
    for variation in name_variations:
        if variation in team_mappings:
            return team_mappings[variation]
    
    # Try a simple fuzzy match on names
    for name in team_mappings:
        if (name in cleaned_name) or (cleaned_name in name):
            return team_mappings[name]
    
    # Return None if no match is found
    return None

def fetch_ncaa_tournament_games(api_key, start_date="2025-03-19", end_date="2025-03-23"):
    """
    Fetch NCAA Tournament games with moneyline and spread odds
    
    Parameters:
    - api_key: API key for sportsgameodds.com
    - start_date: Start date in YYYY-MM-DD format
    - end_date: End date in YYYY-MM-DD format
    
    Returns:
    - List of NCAA Tournament games with odds
    """
    # Base URL for the API
    base_url = "https://api.sportsgameodds.com/v2/events"
    
    # Initialize variables for pagination
    next_cursor = None
    all_games = []
    page_num = 1
    limit = 50  # Number of games per request
    
    print(f"Fetching NCAA Tournament games from {start_date} to {end_date}...")
    
    # Loop through pages until we've fetched all games
    while True:
        # Prepare query parameters
        params = {
            "sportID": "BASKETBALL",
            "leagueID": "NCAAB",
            "oddsAvailable": "true",
            "includeOpposingOdds": "true",
            "startsAfter": start_date,
            "startsBefore": end_date,
            "bookmakerID": 'betmgm',
            "limit": limit
        }
        
        # Add cursor for pagination if we have one
        if next_cursor:
            params["cursor"] = next_cursor
            
        headers = {
            "X-API-KEY": api_key
        }
        
        try:
            # Get games for the specified date range
            print(f"Fetching page {page_num}...")
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Get the games from this page
            games = result.get("data", [])
            all_games.extend(games)
            
            # Check if we have a next cursor
            next_cursor = result.get("nextCursor")
            
            # If no next cursor, we've fetched all games
            if not next_cursor:
                break
                
            # Increment page number and add a small delay
            page_num += 1
            time.sleep(0.5)
            
        except requests.RequestException as e:
            print(f"Error fetching games: {str(e)}")
            break
    
    print(f"Found {len(all_games)} NCAA games for the specified date range")
    return all_games

def extract_odds_data(games, team_mappings=None):
    """
    Extract team names and odds data from the games
    
    Parameters:
    - games: List of game objects from the API
    - team_mappings: Dictionary mapping team names to team IDs
    
    Returns:
    - Two DataFrames: moneyline_df and spread_df with structured odds data
    """
    moneyline_rows = []
    spread_rows = []
    moneyline_count = 0
    spread_count = 0
    games_with_odds = 0
    
    # Load team mappings if not provided
    if team_mappings is None:
        team_mappings = load_team_mappings()
        print(f"Loaded {len(team_mappings)} team name variations for mapping")
    
    for game in games:
        game_id = game.get("eventID")
        
        # Get game start time
        status = game.get("status", {})
        starts_at = status.get("startsAt", "")
        game_date = starts_at.split("T")[0] if starts_at and "T" in starts_at else ""
        game_time = starts_at.split("T")[1].split(".")[0] if starts_at and "T" in starts_at else ""
        
        # Get team info
        teams = game.get("teams", {})
        home_team = teams.get("home", {}).get("names", {}).get("long", "Unknown")
        away_team = teams.get("away", {}).get("names", {}).get("long", "Unknown")
        
        # Match team names to IDs
        home_team_id = match_team_name(home_team, team_mappings)
        away_team_id = match_team_name(away_team, team_mappings)
        
        # Get odds
        odds = game.get("odds", {})
        
        # Log available odds keys for debugging
        if odds and 'NCAAB' in game.get("leagueID", ""):
            print(f"Game: {home_team} vs {away_team} - Available odds keys: {list(odds.keys())}")
            games_with_odds += 1
        
        # Process moneyline odds for home team
        home_ml_key = "points-home-game-ml-home"
        if home_ml_key in odds:
            home_ml_odds = odds[home_ml_key]
            for bookmaker, book_data in home_ml_odds.get("byBookmaker", {}).items():
                if book_data.get("available"):
                    moneyline_count += 1
                    moneyline_rows.append({
                        'game_id': game_id,
                        'game_date': game_date,
                        'game_time': game_time,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_team_id': home_team_id,
                        'away_team_id': away_team_id,
                        'team': home_team,
                        'team_id': home_team_id,
                        'team_type': 'home',
                        'price': book_data.get("odds"),
                        'sportsbook': bookmaker,
                        'last_updated': book_data.get("lastUpdatedAt", ""),
                        'fetch_date': datetime.now().strftime("%Y-%m-%d"),
                        'fetch_time': datetime.now().strftime("%H:%M:%S")
                    })
        
        # Process moneyline odds for away team
        away_ml_key = "points-away-game-ml-away"
        if away_ml_key in odds:
            away_ml_odds = odds[away_ml_key]
            for bookmaker, book_data in away_ml_odds.get("byBookmaker", {}).items():
                if book_data.get("available"):
                    moneyline_count += 1
                    moneyline_rows.append({
                        'game_id': game_id,
                        'game_date': game_date,
                        'game_time': game_time,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_team_id': home_team_id,
                        'away_team_id': away_team_id,
                        'team': away_team,
                        'team_id': away_team_id,
                        'team_type': 'away',
                        'price': book_data.get("odds"),
                        'sportsbook': bookmaker,
                        'last_updated': book_data.get("lastUpdatedAt", ""),
                        'fetch_date': datetime.now().strftime("%Y-%m-%d"),
                        'fetch_time': datetime.now().strftime("%H:%M:%S")
                    })
        
        # Process spread odds for home team using the specific key
        home_sp_key = "points-home-game-sp-home"
        if home_sp_key in odds:
            home_spread_odds = odds[home_sp_key]
            for bookmaker, book_data in home_spread_odds.get("byBookmaker", {}).items():
                if book_data.get("available"):
                    # Get the spread directly from the bookmaker data
                    spread_value = book_data.get("spread")
                    if spread_value is None:
                        # Fallback to line if spread is not available
                        spread_value = home_spread_odds.get("line")
                    
                    if spread_value is not None:
                        spread_count += 1
                        spread_rows.append({
                            'game_id': game_id,
                            'game_date': game_date,
                            'game_time': game_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_team_id': home_team_id,
                            'away_team_id': away_team_id,
                            'team': home_team,
                            'team_id': home_team_id,
                            'team_type': 'home',
                            'spread': spread_value,
                            'price': book_data.get("odds"),
                            'sportsbook': bookmaker,
                            'last_updated': book_data.get("lastUpdatedAt", ""),
                            'fetch_date': datetime.now().strftime("%Y-%m-%d"),
                            'fetch_time': datetime.now().strftime("%H:%M:%S")
                        })
        
        # Process spread odds for away team using the specific key
        away_sp_key = "points-away-game-sp-away"
        if away_sp_key in odds:
            away_spread_odds = odds[away_sp_key]
            for bookmaker, book_data in away_spread_odds.get("byBookmaker", {}).items():
                if book_data.get("available"):
                    # Get the spread directly from the bookmaker data
                    spread_value = book_data.get("spread")
                    if spread_value is None:
                        # Fallback to line if spread is not available
                        spread_value = away_spread_odds.get("line")
                    
                    if spread_value is not None:
                        spread_count += 1
                        spread_rows.append({
                            'game_id': game_id,
                            'game_date': game_date,
                            'game_time': game_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_team_id': home_team_id,
                            'away_team_id': away_team_id,
                            'team': away_team,
                            'team_id': away_team_id,
                            'team_type': 'away',
                            'spread': spread_value,
                            'price': book_data.get("odds"),
                            'sportsbook': bookmaker,
                            'last_updated': book_data.get("lastUpdatedAt", ""),
                            'fetch_date': datetime.now().strftime("%Y-%m-%d"),
                            'fetch_time': datetime.now().strftime("%H:%M:%S")
                        })
    
    # Print statistics
    print(f"\nData Collection Results:")
    print(f"  Games with odds data: {games_with_odds}")
    print(f"  Moneyline entries collected: {moneyline_count}")
    print(f"  Spread entries collected: {spread_count}")
    print(f"  Total entries: {moneyline_count + spread_count}")
    
    # Create DataFrames and sort by game date/time
    moneyline_df = pd.DataFrame(moneyline_rows)
    spread_df = pd.DataFrame(spread_rows)
    
    if not moneyline_df.empty and 'game_date' in moneyline_df.columns and 'game_time' in moneyline_df.columns:
        moneyline_df = moneyline_df.sort_values(['game_date', 'game_time'])
    
    if not spread_df.empty and 'game_date' in spread_df.columns and 'game_time' in spread_df.columns:
        spread_df = spread_df.sort_values(['game_date', 'game_time'])
    
    # Report on team mapping success rate
    if team_mappings:
        all_teams = set()
        mapped_teams = set()
        unmapped_teams = set()
        
        # Process moneyline teams
        if not moneyline_df.empty:
            for team, team_id in zip(moneyline_df['team'], moneyline_df['team_id']):
                all_teams.add(team)
                if pd.notna(team_id):
                    mapped_teams.add(team)
                else:
                    unmapped_teams.add(team)
        
        # Process spread teams
        if not spread_df.empty:
            for team, team_id in zip(spread_df['team'], spread_df['team_id']):
                all_teams.add(team)
                if pd.notna(team_id):
                    mapped_teams.add(team)
                else:
                    unmapped_teams.add(team)
        
        if all_teams:
            print(f"\nTeam Mapping Results:")
            print(f"  Mapped {len(mapped_teams)} out of {len(all_teams)} unique teams ({len(mapped_teams)/len(all_teams)*100:.1f}%)")
            
            if unmapped_teams:
                print(f"  Unmapped teams ({len(unmapped_teams)}):")
                for team in sorted(unmapped_teams):
                    print(f"    - {team}")
        
    return moneyline_df, spread_df

def main():
    parser = argparse.ArgumentParser(description='Fetch NCAA Tournament basketball betting odds')
    parser.add_argument('--api-key', type=str, help='API key for sportsgameodds.com')
    parser.add_argument('--output-dir', type=str, default='../pre_tourney_data', help='Directory to save the output CSV')
    parser.add_argument('--start-date', type=str, default='2025-03-19', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default='2025-03-22', help='End date in YYYY-MM-DD format')
    parser.add_argument('--teams-file', type=str, default='../raw_data/MTeams.csv', help='Path to the MTeams.csv file')
    parser.add_argument('--mapping-file', type=str, default='../pre_tourney_data/mapping.json', help='Path to the mapping.json file')
    
    args = parser.parse_args()
    
    # Check if API key is provided
    api_key = args.api_key
    if not api_key:
        # Try to get it from environment variable
        api_key = os.environ.get('SPORTSGAMEODDS_API_KEY')
        if not api_key:
            print("Error: API key is required. Provide it with --api-key or set SPORTSGAMEODDS_API_KEY environment variable.")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set date range for fetching NCAA tournament games
    start_date = args.start_date
    end_date = args.end_date
    
    print(f"Will fetch NCAA Tournament odds from {start_date} to {end_date}")
    
    # Load team mappings
    print(f"Loading team mappings from {args.teams_file} and {args.mapping_file}...")
    team_mappings = load_team_mappings(args.teams_file, args.mapping_file)
    
    # Fetch NCAA tournament games with odds
    games = fetch_ncaa_tournament_games(api_key, start_date, end_date)
    
    if not games:
        print("No NCAA Tournament games found for the specified date range.")
        sys.exit(0)
    
    # Extract odds data
    print("Extracting odds data...")
    moneyline_df, spread_df = extract_odds_data(games, team_mappings)
    
    if moneyline_df.empty and spread_df.empty:
        print("No valid odds data to save.")
        sys.exit(0)
    
    # Generate date range string for filenames
    date_range = f"{start_date}_to_{end_date}"
    
    # Save moneyline data
    if not moneyline_df.empty:
        moneyline_file = os.path.join(output_dir, f"ncaa_tournament_moneyline_{date_range}.csv")
        moneyline_latest = os.path.join(output_dir, "ncaa_tournament_moneyline_latest.csv")
        moneyline_df.to_csv(moneyline_file, index=False)
        moneyline_df.to_csv(moneyline_latest, index=False)
        print(f"\nSaved moneyline odds data to {moneyline_file}")
        print(f"Also saved to {moneyline_latest} for easy access")
    else:
        print("\nNo moneyline odds data to save.")
    
    # Save spread data
    if not spread_df.empty:
        spread_file = os.path.join(output_dir, f"ncaa_tournament_spread_{date_range}.csv")
        spread_latest = os.path.join(output_dir, "ncaa_tournament_spread_latest.csv")
        spread_df.to_csv(spread_file, index=False)
        spread_df.to_csv(spread_latest, index=False)
        print(f"\nSaved spread odds data to {spread_file}")
        print(f"Also saved to {spread_latest} for easy access")
    else:
        print("\nNo spread odds data to save.")
    
    # Save combined data for backward compatibility
    if not moneyline_df.empty or not spread_df.empty:
        # Add market_type column to each dataframe
        if not moneyline_df.empty:
            moneyline_df['market_type'] = 'moneyline'
        if not spread_df.empty:
            spread_df['market_type'] = 'spread'
            # Rename spread to handicap for backward compatibility
            spread_df = spread_df.rename(columns={'spread': 'handicap'})
        
        # Combine the dataframes
        combined_df = pd.concat([moneyline_df, spread_df], ignore_index=True)
        
        # Save combined data
        combined_file = os.path.join(output_dir, f"ncaa_tournament_odds_{date_range}.csv")
        combined_latest = os.path.join(output_dir, "ncaa_tournament_odds_latest.csv")
        combined_df.to_csv(combined_file, index=False)
        combined_df.to_csv(combined_latest, index=False)
        print(f"\nSaved combined odds data to {combined_file}")
        print(f"Also saved to {combined_latest} for backward compatibility")
    
    # Display summary statistics
    print(f"\n=== Summary ===")
    print(f"Moneyline entries: {len(moneyline_df) if not moneyline_df.empty else 0}")
    print(f"Spread entries: {len(spread_df) if not spread_df.empty else 0}")
    print(f"Total entries: {len(moneyline_df) if not moneyline_df.empty else 0 + len(spread_df) if not spread_df.empty else 0}")
    print(f"Unique games: {len(set(moneyline_df['game_id'].unique()).union(set(spread_df['game_id'].unique() if not spread_df.empty else [])))}")

if __name__ == "__main__":
    main()