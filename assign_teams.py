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

# Get unique player IDs and their names
unique_players = {}
for frame_data in tracking_data.values():
    for player in frame_data:
        track_id = player.get('track_id')
        if track_id is not None and track_id not in unique_players:
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

# Load previous team names if available
previous_team_names = {}
if os.path.exists('outputs/team_names.json'):
    try:
        with open('outputs/team_names.json', 'r') as f:
            previous_team_names = json.load(f)
        print(f"Previous teams: {previous_team_names.get('team1', 'N/A')} vs {previous_team_names.get('team2', 'N/A')}")
        print()
    except (FileNotFoundError, json.JSONDecodeError):
        pass

team1 = input("Team 1 name (e.g., 'Red Team'): ").strip()
if not team1:
    team1 = previous_team_names.get('team1', 'Team 1')
    print(f"  Using: {team1}")

team2 = input("Team 2 name (e.g., 'Yellow Team'): ").strip()
if not team2:
    team2 = previous_team_names.get('team2', 'Team 2')
    print(f"  Using: {team2}")

referee_team = "Referee"

# Save team names
team_names = {
    'team1': team1,
    'team2': team2,
    'referee': referee_team
}
with open('outputs/team_names.json', 'w') as f:
    json.dump(team_names, f, indent=2)

print()
print(f"✓ Teams: {team1} vs {team2}")
print()

# Assign players to teams
print("=" * 60)
print("STEP 2: ASSIGN PLAYERS TO TEAMS")
print("=" * 60)
print()
print("For each player, enter:")
print(f"  1 = {team1}")
print(f"  2 = {team2}")
print(f"  R = {referee_team}")
print("  ENTER = keep previous assignment")
print()

for track_id in sorted(unique_players.keys()):
    player_name = unique_players[track_id]

    # Show previous assignment if exists
    previous_team = team_assignments.get(track_id, "")
    if previous_team:
        prompt = f"{player_name} (ID {track_id}) - Current: '{previous_team}' [1/{team1[0]}=Team1, 2/{team2[0]}=Team2, R=Ref, ENTER=keep]: "
    else:
        prompt = f"{player_name} (ID {track_id}) - Assign team [1/{team1[0]}=Team1, 2/{team2[0]}=Team2, R=Ref]: "

    assignment = input(prompt).strip().upper()

    if assignment in ['1', team1[0].upper()]:
        team_assignments[track_id] = team1
        print(f"  ✓ {player_name} → {team1}")
    elif assignment in ['2', team2[0].upper()]:
        team_assignments[track_id] = team2
        print(f"  ✓ {player_name} → {team2}")
    elif assignment == 'R':
        team_assignments[track_id] = referee_team
        print(f"  ✓ {player_name} → {referee_team}")
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
    player_name = unique_players[track_id]
    team_counts[team].append(player_name)

for team in [team1, team2, referee_team, "Unknown"]:
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
print("instead of generic Team 0/Team 1")
print()
