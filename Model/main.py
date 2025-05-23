import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

# -------------------------------------------------
# 1. Load and Clean Data
# -------------------------------------------------
file_path = r'C:\Users\callo\OneDrive\Desktop\CSC 217 Project\Dataset\cbb.csv'
df = pd.read_csv(file_path)

# Clean column headers by stripping whitespace
df.columns = df.columns.str.strip()

# Drop rows where SEED is missing
df = df.dropna(subset=['SEED'])

# Fill other missing values
def clean_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

clean_data(df)

# Team name mapping to match cbb.csv dataset names
TEAM_NAME_MAPPING = {
    'Ole Miss': 'Mississippi',
    'SDSU/UNC': 'North Carolina',  # Taking UNC for play-in
    'AU/Mount': 'American',  # Taking American for play-in
    'Saint Mary\'s': 'Saint Marys',
    'UConn': 'Connecticut',
    'UNC Wilmington': 'UNC-Wilmington',
    'Norfolk St': 'Norfolk State',
    'Colorado St': 'Colorado State',
    'Alabama St': 'Alabama State',
    'St. John\'s': 'St Johns'
}

# Updated 2025 NCAA Tournament Bracket
BRACKET_2025 = {
    'South': [
        ('Auburn', 1), ('Alabama State', 16),
        ('Louisville', 8), ('Creighton', 9),
        ('Michigan', 5), ('UC San Diego', 12),
        ('Texas A&M', 4), ('Yale', 13),
        ('Mississippi', 6), ('North Carolina', 11),
        ('Iowa State', 3), ('Lipscomb', 14),
        ('Marquette', 7), ('New Mexico', 10),
        ('Michigan State', 2), ('Bryant', 15)
    ],
    'East': [
        ('Duke', 1), ('American', 16),
        ('Mississippi State', 8), ('Baylor', 9),
        ('Oregon', 5), ('Liberty', 12),
        ('Arizona', 4), ('Akron', 13),
        ('BYU', 6), ('VCU', 11),
        ('Wisconsin', 3), ('Montana', 14),
        ('Saint Marys', 7), ('Vanderbilt', 10),
        ('Alabama', 2), ('Robert Morris', 15)
    ],
    'Midwest': [
        ('Houston', 1), ('SIU Edwardsville', 16),
        ('Gonzaga', 8), ('Georgia', 9),
        ('Clemson', 5), ('McNeese', 12),
        ('Purdue', 4), ('High Point', 13),
        ('Illinois', 6), ('Xavier', 11),
        ('Kentucky', 3), ('Troy', 14),
        ('UCLA', 7), ('Utah State', 10),
        ('Tennessee', 2), ('Wofford', 15)
    ],
    'West': [
        ('Florida', 1), ('Norfolk State', 16),
        ('Connecticut', 8), ('Oklahoma', 9),
        ('Memphis', 5), ('Colorado State', 12),
        ('Maryland', 4), ('Grand Canyon', 13),
        ('Missouri', 6), ('Drake', 11),
        ('Texas Tech', 3), ('UNC-Wilmington', 14),
        ('Kansas', 7), ('Arkansas', 10),
        ('St Johns', 2), ('Omaha', 15)
    ]
}

# Create more comprehensive rating based on team stats
def calculate_team_rating(row):
    # Offensive rating components
    offensive = (
        row['ADJOE'] * 0.3 +  # Adjusted offensive efficiency
        row['EFG_O'] * 0.2 +  # Effective field goal percentage
        (100 - row['TOR']) * 0.1  # Turnover rate (inverted)
    )
    
    # Defensive rating components
    defensive = (
        (100 - row['ADJDE']) * 0.3 +  # Adjusted defensive efficiency (inverted)
        (100 - row['EFG_D']) * 0.2 +  # Defensive effective field goal percentage (inverted)
        row['TORD'] * 0.1  # Defensive turnover rate
    )
    
    # Seed importance increases as tournament progresses
    seed_factor = (16 - row['SEED']) * 0.1  # Better seeds get higher ratings
    
    return offensive + defensive + seed_factor

df['RATING'] = df.apply(calculate_team_rating, axis=1)

# -------------------------------------------------
# 2. Create the WIN column for "Sweet 16 or better"
# -------------------------------------------------
valid_rounds_for_win = ['S16','E8','F4','2ND','Champ']
df['WIN'] = df['POSTSEASON'].isin(valid_rounds_for_win).astype(int)

# -------------------------------------------------
# 3. Feature Selection
# -------------------------------------------------
# Update feature selection to include only available columns
features = [
    'ADJOE', 'ADJDE',    # Adjusted efficiency
    'EFG_O', 'EFG_D',    # Shooting effectiveness
    'TOR', 'TORD',       # Turnover rates
    'WAB',               # Wins Above Bubble
    'BARTHAG',           # Power rating
    'SEED'               # Tournament seed
]

# Print available columns to verify
print("\nAvailable columns in dataset:")
print(df.columns.tolist())

# Verify all features exist
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing features in dataset: {missing_features}")

X = df[features]
y = df['WIN']

# -------------------------------------------------
# 4. Train/Test Split and Scaling
#    (Using stratify=y so both sets contain 0/1)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # <--- IMPORTANT: ensures distribution of 0/1 is maintained
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# 5. Train the XGBoost Model
# -------------------------------------------------
model = XGBClassifier(
    eval_metric='logloss',
    base_score=0.5,   # must be in (0,1) for logistic loss
    random_state=42
)
model.fit(X_train_scaled, y_train)

# -------------------------------------------------
# 6. Evaluate the Model
# -------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

# If the test set has only one class (all 0 or all 1),
# we can't compute log_loss. Check and handle gracefully:
unique_labels = set(y_test)
if len(unique_labels) == 1:
    print(f"\nAll labels in y_test are the same ({unique_labels}). Log loss is not defined.")
    loss_message = "N/A (single label in test set)"
else:
    loss = log_loss(y_test, y_pred_prob)
    loss_message = f"{loss:.4f}"

print(f"\nXGBoost Model Accuracy: {accuracy:.2f}")
print(f"Log Loss: {loss_message}")

# -------------------------------------------------
# 7. Prepare the Current Tournament Teams
# -------------------------------------------------
# Example: sort by SEED and take top 64
current_tourney_teams = df.sort_values(by='SEED').head(64).copy()
current_tourney_teams.reset_index(drop=True, inplace=True)

# Compute each team's "Sweet16 probability" (model’s prob for y=1).
team_features_scaled = scaler.transform(current_tourney_teams[features])
current_tourney_teams['Sweet16_Prob'] = model.predict_proba(team_features_scaled)[:, 1]

# Keep just what we need for bracket simulation
tourney_df = current_tourney_teams[['TEAM','SEED','Sweet16_Prob']].copy()

# Setup proper tournament bracket structure
def create_tournament_bracket(df):
    """Create tournament bracket using predefined 2025 teams"""
    bracket = []
    
    # Create a mapping of standardized team names
    team_mapping = {**TEAM_NAME_MAPPING}
    
    for region, teams in BRACKET_2025.items():
        for team, seed in teams:
            # Get standardized team name
            std_team = team_mapping.get(team, team)
            
            # Get team stats from dataset
            team_data = df[df['TEAM'].str.lower() == std_team.lower()]
            if len(team_data) == 0:
                print(f"Warning: No data found for {std_team}, using average stats for seed {seed}")
                rating = df[df['SEED'] == seed]['RATING'].mean()
            else:
                rating = team_data.iloc[0]['RATING']
            
            bracket.append({
                'team': std_team,
                'seed': seed,
                'rating': rating,
                'region': region
            })
    
    return pd.DataFrame(bracket)

def calculate_win_probability(teamA, teamB):
    """
    Calculate win probability using both rating difference and seed difference
    """
    rating_diff = teamA['rating'] - teamB['rating']
    seed_diff = teamB['seed'] - teamA['seed']  # Reverse because lower seed is better
    
    # Convert rating difference to probability using logistic function
    rating_factor = 1 / (1 + np.exp(-rating_diff / 100))
    
    # Seed difference factor (gives slight advantage to better seeds)
    seed_factor = 0.5 + (seed_diff * 0.01)  # Small adjustment based on seed difference
    
    # Combine factors (70% rating, 30% seed)
    final_prob = (rating_factor * 0.7) + (seed_factor * 0.3)
    
    # Ensure probability is between 0.05 and 0.95 to avoid extreme values
    return np.clip(final_prob, 0.05, 0.95)

def simulate_matchup(teamA, teamB):
    prob = calculate_win_probability(teamA, teamB)
    return (teamA, prob) if np.random.random() < prob else (teamB, 1-prob)

def simulate_round(teams, round_name):
    winners = []
    matchups = []
    
    for i in range(0, len(teams), 2):
        teamA = teams.iloc[i]
        teamB = teams.iloc[i+1]
        prob_A_wins = calculate_win_probability(teamA, teamB)
        prob_B_wins = 1 - prob_A_wins
        
        # Simulate the winner based on probabilities
        if np.random.random() < prob_A_wins:
            winner = teamA
            win_prob = prob_A_wins
        else:
            winner = teamB
            win_prob = prob_B_wins
            
        matchups.append({
            'round': round_name,
            'teamA': teamA['team'],
            'seedA': teamA['seed'],
            'teamB': teamB['team'],
            'seedB': teamB['seed'],
            'winner': winner['team'],
            'probability': win_prob,
            'probA': prob_A_wins,
            'probB': prob_B_wins
        })
        winners.append(winner)
    
    return pd.DataFrame(winners), pd.DataFrame(matchups)

def simulate_tournament(bracket_df):
    """Simulate entire tournament with proper regional progression"""
    rounds = [
        ("Round of 64", 32),
        ("Round of 32", 16),
        ("Sweet 16", 8),
        ("Elite 8", 4),
        ("Final Four", 2),
        ("Championship", 1)
    ]
    
    current_teams = bracket_df.copy()
    all_matchups = []
    
    # For first 4 rounds, keep regions separate
    for round_name, num_winners in rounds[:4]:
        regional_winners = []
        print(f"\n{round_name}:")
        
        # Process each region separately
        for region in ['East', 'West', 'South', 'Midwest']:
            regional_teams = current_teams[current_teams['region'] == region]
            winners, matchups = simulate_round(regional_teams, round_name)
            regional_winners.append(winners)
            all_matchups.append(matchups)
            
            # Print regional results
            print(f"\n{region} Region:")
            for _, match in matchups.iterrows():
                print(f"{match['teamA']} ({match['seedA']}) [{match['probA']:.3f}] vs "
                      f"{match['teamB']} ({match['seedB']}) [{match['probB']:.3f}] "
                      f"-> Winner: {match['winner']}")
        
        current_teams = pd.concat(regional_winners)
    
    # Final Four and Championship (no regions)
    for round_name, num_winners in rounds[4:]:
        print(f"\n{round_name}:")
        winners, matchups = simulate_round(current_teams, round_name)
        current_teams = winners
        all_matchups.append(matchups)
        
        for _, match in matchups.iterrows():
            print(f"{match['teamA']} ({match['seedA']}) [{match['probA']:.3f}] vs "
                  f"{match['teamB']} ({match['seedB']}) [{match['probB']:.3f}] "
                  f"-> Winner: {match['winner']}")
    
    return pd.concat(all_matchups), current_teams.iloc[0]

# Create and simulate tournament
tourney_bracket = create_tournament_bracket(df)
matchups_df, champion = simulate_tournament(tourney_bracket)

# Save results and display champion
matchups_df.to_csv('tournament_results.csv', index=False)
print("\n=== PREDICTED CHAMPION ===")
print(f"{champion['team']} (Seed {champion['seed']})")
