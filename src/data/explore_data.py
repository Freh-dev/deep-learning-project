import os
import json
import sys
from pathlib import Path

# Get current directory
repo_path = Path.cwd()
print(f"📁 Repository: {repo_path}\n")

# ============================================================
# STEP 1: Find all JSON files
# ============================================================
print("=" * 60)
print("STEP 1: Finding all JSON files")
print("=" * 60)

json_files = list(repo_path.rglob("*.json"))
print(f"Found {len(json_files)} JSON files:")

# Group by location
for f in json_files[:20]:  # Show first 20
    rel_path = f.relative_to(repo_path)
    size_kb = f.stat().st_size / 1024
    print(f"  - {rel_path} ({size_kb:.1f} KB)")

if len(json_files) > 20:
    print(f"  ... and {len(json_files) - 20} more")

# ============================================================
# STEP 2: Check for known data files
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Looking for benchmark data files")
print("=" * 60)

possible_files = ['scenarios.json', 'qa_pairs.json', 'benchmark.json', 'data.json']

for filename in possible_files:
    found = list(repo_path.rglob(filename))
    if found:
        f = found[0]
        size_kb = f.stat().st_size / 1024
        print(f"✅ Found: {f.relative_to(repo_path)} ({size_kb:.1f} KB)")
    else:
        print(f"❌ Not found: {filename}")

# ============================================================
# STEP 3: Inspect the first found scenarios file
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Inspecting the scenarios file")
print("=" * 60)

scenario_candidates = list(repo_path.rglob('scenarios.json'))
if scenario_candidates:
    scenario_path = scenario_candidates[0]
    print(f"📄 Inspecting: {scenario_path.relative_to(repo_path)}")
    
    with open(scenario_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    
    print(f"\nData type: {type(content).__name__}")
    
    if isinstance(content, list):
        print(f"Number of items: {len(content)}")
        if len(content) > 0:
            print(f"\nFirst item keys: {list(content[0].keys())}")
            print(f"\nFirst item sample:")
            print(json.dumps(content[0], indent=2)[:800] + "...\n")
            
            # Check for risk label
            print("Available fields in first item:")
            for key in content[0].keys():
                value = content[0][key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  - {key}: {value[:80]}...")
                else:
                    print(f"  - {key}: {value}")
                
    elif isinstance(content, dict):
        print(f"Top-level keys: {list(content.keys())}")
        for key, value in content.items():
            if isinstance(value, list):
                print(f"  - {key}: list of {len(value)} items")
                if len(value) > 0:
                    print(f"    First item keys: {list(value[0].keys()) if isinstance(value[0], dict) else type(value[0])}")
            else:
                print(f"  - {key}: {type(value).__name__}")

# ============================================================
# STEP 4: Check mkdata folder for data generation
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Checking mkdata folder for generated data")
print("=" * 60)

mkdata_path = repo_path / 'mkdata'
if mkdata_path.exists():
    print(f"📂 mkdata contains:")
    for f in sorted(mkdata_path.iterdir()):
        if f.is_file():
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
else:
    print("❌ mkdata folder not found")

print("\n" + "=" * 60)
print("Done! Let me know what you found.")
print("=" * 60)