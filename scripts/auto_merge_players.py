#!/usr/bin/env python3
"""
Auto-merge duplicate player IDs and remove public/bench people.
"""

import sys
import os
import json
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import get_config

config = get_config()
output_dir = config.get_output_dir()

print("=" * 70)
print("AUTO-MERGE JUGADORES DUPLICADOS")
print("=" * 70)
print()

# Load player names
player_names_file = f"{output_dir}/player_names.json"
with open(player_names_file, 'r') as f:
    player_names_raw = json.load(f)
    player_names = {int(k): v for k, v in player_names_raw.items()}

print(f"[+] Nombres actuales: {len(player_names)} IDs")
print()

# Group by name
name_to_ids = defaultdict(list)
for track_id, name in player_names.items():
    name_to_ids[name].append(track_id)

print("Jugadores agrupados por nombre:")
print()
for name, ids in sorted(name_to_ids.items()):
    if len(ids) > 1:
        print(f"  '{name}': {len(ids)} IDs -> {sorted(ids)}")
    else:
        print(f"  '{name}': ID {ids[0]}")
print()

# Identify public/bench IDs to remove
public_ids = name_to_ids.get('public', [])
bench_ids = name_to_ids.get('bench', [])
remove_ids = set(public_ids + bench_ids)

if remove_ids:
    print(f"[!] IDs a ELIMINAR (público/banquillo): {sorted(remove_ids)}")
    print()

# Create merge mapping
id_mapping = {}
merged_names = {}

for name, ids in name_to_ids.items():
    # Skip public/bench
    if name.lower() in ['public', 'bench']:
        continue

    if len(ids) > 1:
        # Merge to first ID
        target_id = sorted(ids)[0]
        merged_names[target_id] = name

        for source_id in ids[1:]:
            id_mapping[source_id] = target_id

        print(f"[M] Mergeando '{name}': {sorted(ids)} -> ID {target_id}")
    else:
        # Single ID, keep as is
        merged_names[ids[0]] = name

print()
print(f"[+] Total merges: {len(id_mapping)}")
print(f"[+] IDs finales: {len(merged_names)} jugadores")
print()

# Load tracking data
tracking_file = f"{output_dir}/tracked_players_filtered.json"
with open(tracking_file, 'r') as f:
    tracking_data = json.load(f)

print(f"[+] Cargados datos de tracking: {len(tracking_data)} frames")

# Apply merges and removals
cleaned_data = {}
removed_count = 0
merged_count = 0

for frame_idx, players in tracking_data.items():
    cleaned_players = []

    for player in players:
        old_id = player.get('track_id')

        # Remove public/bench
        if old_id in remove_ids:
            removed_count += 1
            continue

        # Apply merge mapping
        if old_id in id_mapping:
            new_id = id_mapping[old_id]
            player['track_id'] = new_id
            merged_count += 1
        else:
            new_id = old_id

        # Add name
        if new_id in merged_names:
            player['name'] = merged_names[new_id]
        else:
            player['name'] = f"Player {new_id}"

        cleaned_players.append(player)

    if cleaned_players:
        cleaned_data[frame_idx] = cleaned_players

print(f"[+] Detecciones de público eliminadas: {removed_count}")
print(f"[+] Detecciones mergeadas: {merged_count}")
print()

# Get final unique IDs
unique_ids = set()
for players in cleaned_data.values():
    for player in players:
        unique_ids.add(player['track_id'])

print("=" * 70)
print("RESULTADO FINAL")
print("=" * 70)
print()
print(f"IDs únicos: {len(unique_ids)}")
print()
print("Roster final:")
for track_id in sorted(unique_ids):
    name = merged_names.get(track_id, f"Player {track_id}")
    print(f"  ID {track_id:2d}: {name}")
print()

# Save cleaned data
output_file = f"{output_dir}/tracked_players_named.json"
with open(output_file, 'w') as f:
    json.dump(cleaned_data, f, indent=2)

print(f"[+] Datos limpios guardados en: {output_file}")

# Save merged player names
with open(player_names_file, 'w') as f:
    json.dump(merged_names, f, indent=2)

print(f"[+] Nombres actualizados en: {player_names_file}")
print()

print("=" * 70)
print("COMPLETADO!")
print("=" * 70)
print()
print("Siguiente paso:")
print("  python scripts/assign_teams.py")
print()
