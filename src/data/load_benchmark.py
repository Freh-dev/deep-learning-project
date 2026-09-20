from src.data.preprocess import load_scenarios, split_scenarios


if __name__ == "__main__":
    scenarios = load_scenarios()
    splits = split_scenarios(scenarios)
    print(f"Loaded {len(scenarios)} validated scenarios.")
    print(scenarios["risk_level"].value_counts().sort_index())
    for name, split in splits.items():
        print(f"{name}: {len(split)} rows")