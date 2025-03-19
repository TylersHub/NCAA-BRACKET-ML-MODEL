import pandas as pd
import graphviz

def draw_bracket(results_file):
    # Read tournament results
    df = pd.read_csv(results_file)
    
    # Create new digraph
    dot = graphviz.Digraph(comment='NCAA Tournament Bracket')
    dot.attr(rankdir='LR')  # Left to right direction
    
    # Style settings
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white')
    
    # Track nodes by round for proper ranking
    round_nodes = {
        "Round of 64": [],
        "Round of 32": [],
        "Sweet 16": [],
        "Elite 8": [],
        "Final Four": [],
        "Championship": []
    }
    
    # Process each round
    for round_name in round_nodes.keys():
        round_matches = df[df['round'] == round_name]
        
        for _, match in round_matches.iterrows():
            # Create unique IDs for each team in this matchup
            team_a_id = f"{round_name}_{match['teamA']}_{match['seedA']}"
            team_b_id = f"{round_name}_{match['teamB']}_{match['seedB']}"
            winner_id = f"{round_name}_winner_{match['winner']}"
            
            # Create team nodes with seeds and probabilities
            team_a_label = f"({match['seedA']}) {match['teamA']}\n[{match['probA']:.3f}]"
            team_b_label = f"({match['seedB']}) {match['teamB']}\n[{match['probB']:.3f}]"
            
            # Color the winner
            winner_color = 'lightgreen' if match['winner'] == match['teamA'] else 'white'
            dot.node(team_a_id, team_a_label, fillcolor=winner_color)
            winner_color = 'lightgreen' if match['winner'] == match['teamB'] else 'white'
            dot.node(team_b_id, team_b_label, fillcolor=winner_color)
            
            # Add to round nodes for ranking
            round_nodes[round_name].extend([team_a_id, team_b_id])
            
            # Create invisible winner node for proper edge routing
            dot.node(winner_id, "", style='invis')
            dot.edge(team_a_id, winner_id)
            dot.edge(team_b_id, winner_id)
            
            # Connect winner to next round if not championship
            if round_name != "Championship":
                next_round_match = df[
                    (df['teamA'] == match['winner']) | 
                    (df['teamB'] == match['winner'])
                ].iloc[1]  # Get next match for this winner
                next_round_id = f"{next_round_match['round']}_{match['winner']}_" + \
                              str(next_round_match['seedA'] if next_round_match['teamA'] == match['winner'] 
                                  else next_round_match['seedB'])
                dot.edge(winner_id, next_round_id)
    
    # Force ranking for each round
    with dot.subgraph() as s:
        s.attr(rank='same')
        for round_name, nodes in round_nodes.items():
            for node in nodes:
                s.node(node)
    
    # Save the bracket
    dot.render('tournament_bracket', format='png', cleanup=True)

if __name__ == "__main__":
    draw_bracket('tournament_results.csv')
