#!/usr/bin/env python3
"""
Assign teams to tracked players.

This script allows you to assign specific team names to players
instead of generic "Team 0" or "Team 1".
"""

import sys
import os
import json
from collections import defaultdict

print("=" * 60)
print("TEAM ASSIGNMENT")
print("=" * 60)
print()

# Check for required files
tracking_file = None
if os.path.exists("outputs/tracked_players_named.json"):
    tracking_file = "outputs/tracked_players_named.json"
    print("✓ Using named tracking data")
elif os.path.exists("outputs/tracked_players_filtered.json"):
    tracking_file = "outputs/tracked_players_filtered.json"
    print("✓ Using filtered tracking data")
elif os.path.exists("outputs/tracked_players.json"):
    tracking_file = "outputs/tracked_players.json"
    print("✓ Using raw tracking data")
else:
    print("❌ No tracking data found")
    sys.exit(1)

print(f"✓ Tracking data: {tracking_file}")
print()

# Load tracking data
with open(tracking_file, 'r') as f:
    tracking_data = json.load(f)

# Get unique player IDs and their LATEST names (in case names were updated)
unique_players = {}
for frame_data in tracking_data.values():
    for player in frame_data:
        track_id = player.get('track_id')
        if track_id is not None:
            # Always update to latest name (handles name updates from assign_names.py)
            name = player.get('name', f'Player {track_id}')
            unique_players[track_id] = name

print(f"✓ Found {len(unique_players)} unique players")
print()

# Load previous team assignments if available
team_assignments = {}
if os.path.exists('outputs/team_assignments.json'):
    try:
        with open('outputs/team_assignments.json', 'r') as f:
            team_assignments_raw = json.load(f)
            team_assignments = {int(k): v for k, v in team_assignments_raw.items()}
        print(f"✓ Loaded {len(team_assignments)} previous team assignments")
        print()
    except (FileNotFoundError, json.JSONDecodeError):
        pass

# Get team names
print("=" * 60)
print("STEP 1: DEFINE TEAMS")
print("=" * 60)
print()
print("Enter the names of the two teams (e.g., 'Red Team', 'Yellow Team')")
print()

# Color suggestions (BGR format for OpenCV)
COLOR_SUGGESTIONS = {
    'red': ([0, 0, 255], 'Rojo/Red'),
    'blue': ([255, 0, 0], 'Azul/Blue'),
    'green': ([0, 255, 0], 'Verde/Green'),
    'yellow': ([0, 255, 255], 'Amarillo/Yellow'),
    'orange': ([0, 165, 255], 'Naranja/Orange'),
    'purple': ([255, 0, 255], 'Morado/Purple'),
    'cyan': ([255, 255, 0], 'Cian/Cyan'),
    'white': ([255, 255, 255], 'Blanco/White'),
    'black': ([0, 0, 0], 'Negro/Black'),
    'gray': ([128, 128, 128], 'Gris/Gray')
}

# Load previous team names if available
previous_team_names = {}
if os.path.exists('outputs/team_names.json'):
    try:
        with open('outputs/team_names.json', 'r') as f:
            previous_team_names = json.load(f)

        # Handle both old format (string) and new format (dict)
        team1_prev = previous_team_names.get('team1', 'N/A')
        team2_prev = previous_team_names.get('team2', 'N/A')
        if isinstance(team1_prev, dict):
            team1_prev = team1_prev.get('name', 'N/A')
        if isinstance(team2_prev, dict):
            team2_prev = team2_prev.get('name', 'N/A')

        print(f"Previous teams: {team1_prev} vs {team2_prev}")
        print()
    except (FileNotFoundError, json.JSONDecodeError):
        pass

team1 = input("Team 1 name (e.g., 'Red Team'): ").strip()
if not team1:
    prev = previous_team_names.get('team1', 'Team 1')
    team1 = prev.get('name', prev) if isinstance(prev, dict) else prev
    print(f"  Using: {team1}")

# Color selection for team 1
print("\nColors (primary): red, blue, green, yellow, orange")
print("Colors (secondary): purple, cyan, white, black, gray")
team1_color_input = input(f"Team 1 color (e.g., 'red', 'blue'): ").strip().lower()
if team1_color_input in COLOR_SUGGESTIONS:
    team1_color = COLOR_SUGGESTIONS[team1_color_input][0]
    print(f"  ✓ {COLOR_SUGGESTIONS[team1_color_input][1]}")
else:
    # Try to get from previous
    prev = previous_team_names.get('team1', {})
    team1_color = prev.get('color', [0, 0, 255]) if isinstance(prev, dict) else [0, 0, 255]
    print(f"  Using previous/default color")

team2 = input("\nTeam 2 name (e.g., 'Yellow Team'): ").strip()
if not team2:
    prev = previous_team_names.get('team2', 'Team 2')
    team2 = prev.get('name', prev) if isinstance(prev, dict) else prev
    print(f"  Using: {team2}")

# Color selection for team 2
print("\nColors (primary): red, blue, green, yellow, orange")
print("Colors (secondary): purple, cyan, white, black, gray")
team2_color_input = input(f"Team 2 color (e.g., 'yellow', 'green'): ").strip().lower()
if team2_color_input in COLOR_SUGGESTIONS:
    team2_color = COLOR_SUGGESTIONS[team2_color_input][0]
    print(f"  ✓ {COLOR_SUGGESTIONS[team2_color_input][1]}")
else:
    # Try to get from previous
    prev = previous_team_names.get('team2', {})
    team2_color = prev.get('color', [0, 255, 255]) if isinstance(prev, dict) else [0, 255, 255]
    print(f"  Using previous/default color")

referee_team = "Referee"
public_category = "Public"

# Save team names with colors (new format)
team_names = {
    'team1': {
        'name': team1,
        'color': team1_color
    },
    'team2': {
        'name': team2,
        'color': team2_color
    },
    'referee': {
        'name': referee_team,
        'color': [128, 128, 128]  # Gray
    },
    'public': {
        'name': public_category,
        'color': [64, 64, 64]  # Dark gray
    }
}
with open('outputs/team_names.json', 'w') as f:
    json.dump(team_names, f, indent=2)

print()
print(f"✓ Teams: {team1} vs {team2}")
print(f"✓ Other: {referee_team}, {public_category}")
print()

# Assign players to teams
print("=" * 60)
print("STEP 2: ASSIGN PLAYERS TO TEAMS")
print("=" * 60)
print()
print("For each player, enter:")
print(f"  1 = {team1}")
print(f"  2 = {team2}")
print(f"  3 = {referee_team}")
print(f"  P = {public_category} (crowd/bench)")
print("  ENTER = keep previous assignment")
print()

for track_id in sorted(unique_players.keys()):
    player_name = unique_players[track_id]

    # Show previous assignment if exists
    previous_team = team_assignments.get(track_id, "")
    if previous_team:
        prompt = f"{player_name} (ID {track_id}) - Current: '{previous_team}' [1=T1, 2=T2, 3=Ref, P=Public, ENTER=keep]: "
    else:
        prompt = f"{player_name} (ID {track_id}) - Assign [1=T1, 2=T2, 3=Ref, P=Public]: "

    assignment = input(prompt).strip().upper()

    if assignment == '1':
        team_assignments[track_id] = team1
        print(f"  ✓ {player_name} → {team1}")
    elif assignment == '2':
        team_assignments[track_id] = team2
        print(f"  ✓ {player_name} → {team2}")
    elif assignment == '3':
        team_assignments[track_id] = referee_team
        print(f"  ✓ {player_name} → {referee_team}")
    elif assignment == 'P':
        team_assignments[track_id] = public_category
        print(f"  ✓ {player_name} → {public_category}")
    elif previous_team:
        # Keep previous
        print(f"  ✓ {player_name} → {previous_team} (kept)")
    else:
        # No assignment
        team_assignments[track_id] = "Unknown"
        print(f"  ⚠ {player_name} → Unknown (no assignment)")

    print()

# Save team assignments
with open('outputs/team_assignments.json', 'w') as f:
    json.dump(team_assignments, f, indent=2)

print("✓ Team assignments saved to outputs/team_assignments.json")
print()

# Update tracking data with team assignments
print("Applying team assignments to tracking data...")

for frame_data in tracking_data.values():
    for player in frame_data:
        track_id = player.get('track_id')
        if track_id is not None and track_id in team_assignments:
            player['team'] = team_assignments[track_id]

# Save updated tracking data
output_file = tracking_file.replace('.json', '_teams.json')
with open(output_file, 'w') as f:
    json.dump(tracking_data, f, indent=2)

print(f"✓ Updated tracking data saved to {output_file}")
print()

# Show team summary
print("=" * 60)
print("TEAM SUMMARY")
print("=" * 60)
print()

team_counts = defaultdict(list)
for track_id, team in team_assignments.items():
    # Handle case where track_id might not be in unique_players (e.g., after merges)
    player_name = unique_players.get(track_id, f'Player {track_id}')
    team_counts[team].append(player_name)

for team in [team1, team2, referee_team, public_category, "Unknown"]:
    if team in team_counts:
        players = team_counts[team]
        print(f"{team}: {len(players)} players")
        for player_name in players:
            print(f"  - {player_name}")
        print()

print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print()
print("Next step:")
print("  Create video with team names:")
print(f"  python create_annotated_video.py")
print()
print(f"The video will show team names ({team1}, {team2}, {referee_team})")
print(f"Players marked as '{public_category}' will be hidden in the video")
print("instead of generic Team 0/Team 1")
print()
