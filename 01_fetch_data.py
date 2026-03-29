"""
STEP 1: Build the Vehicle Dataset
==================================
This script downloads vehicle make/model data from the free NHTSA
(National Highway Traffic Safety Administration) public API and converts
it into natural language training text.

KEY CONCEPT: Why does data format matter so much?
A language model learns ONLY from the patterns it sees in text.
If we feed it raw CSV data like:
    Ford,Mustang,1964
    Ford,F-150,1975

...it learns to predict CSV formatting, not to answer questions.

Instead we format the SAME data as natural language + Q&A pairs:
    Ford makes the following models: Mustang, F-150, Explorer.
    Q: What models does Ford make?
    A: Ford makes the Mustang, F-150, and Explorer.

Now the model learns to ANSWER QUESTIONS about vehicles. The data is
identical — only the format changes. This is why data engineering is
considered the most important part of building a good LLM.

Run: python 01_fetch_data.py
"""

import requests
import os
import time
import json

# Where we'll save the training text
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "vehicles.txt")

# NHTSA API base URL (free, no API key needed)
NHTSA_BASE = "https://vpic.nhtsa.dot.gov/api"


def fetch_all_makes():
    """
    Fetch every vehicle make registered with NHTSA.
    Returns a list of dicts: [{"MakeId": 440, "MakeName": "FORD"}, ...]
    """
    print("Fetching all vehicle makes from NHTSA API...")
    url = f"{NHTSA_BASE}/vehicles/getallmakes?format=json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    makes = data["Results"]
    print(f"  Found {len(makes)} makes total.")
    return makes


def fetch_models_for_make(make_name):
    """
    Fetch all models for a given make name.
    Returns a list of model name strings.
    """
    url = f"{NHTSA_BASE}/vehicles/getmodelsformake/{requests.utils.quote(make_name)}?format=json"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        models = [r["Model_Name"] for r in data["Results"] if r.get("Model_Name")]
        return list(set(models))  # deduplicate
    except Exception:
        return []


def make_training_text(make_name, models):
    """
    Convert a make + its models into multiple natural language training examples.

    KEY CONCEPT: Data Augmentation
    We write the same fact in several different ways. This is called
    "data augmentation" — it helps the model generalize. A student who
    only ever sees "2 + 2 = 4" might not recognize "What is two plus two?"
    The same is true for our LLM.
    """
    if not models:
        return ""

    lines = []
    make = make_name.strip().title()  # e.g. "FORD" → "Ford"
    model_list = ", ".join(models)

    # --- Declarative sentence ---
    if len(models) == 1:
        lines.append(f"{make} makes the following model: {models[0]}.\n")
    else:
        lines.append(f"{make} makes the following models: {model_list}.\n")

    # --- Q&A: models for a make ---
    if len(models) == 1:
        lines.append(f"Q: What models does {make} make?\nA: {make} makes the {models[0]}.\n")
    elif len(models) <= 5:
        ans = ", ".join(models[:-1]) + f", and {models[-1]}"
        lines.append(f"Q: What models does {make} make?\nA: {make} makes the {ans}.\n")
    else:
        # For makes with many models, list a selection
        selection = models[:8]
        ans = ", ".join(selection[:-1]) + f", and {selection[-1]}"
        lines.append(
            f"Q: What models does {make} make?\n"
            f"A: {make} makes many models including the {ans}.\n"
        )

    # --- Reverse Q&A: who makes a given model ---
    # Do this for the first few models so the file doesn't get enormous
    for model in models[:5]:
        lines.append(f"Q: Who makes the {model}?\nA: The {model} is made by {make}.\n")
        lines.append(f"Q: Is the {model} a {make} vehicle?\nA: Yes, the {model} is made by {make}.\n")

    # --- Membership check ---
    lines.append(f"Q: Does {make} make a vehicle called the {models[0]}?\nA: Yes, {make} makes the {models[0]}.\n")

    return "\n".join(lines) + "\n"


def build_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Get all makes
    all_makes = fetch_all_makes()

    # KEY DECISION: The NHTSA list is alphabetical, so the first 1000 entries
    # are mostly obscure custom manufacturers (names starting with A, B, C...).
    # Well-known brands like Ford, Toyota, GM come much later.
    #
    # Strategy: Explicitly fetch well-known brands FIRST so the model sees
    # them many times, then add the rest of the alphabetical list for breadth.
    #
    # This is an important data curation lesson: the quality and distribution
    # of your training data directly determines what the model learns.

    # Priority makes — the brands most people know about
    PRIORITY_MAKES = [
        "FORD", "TOYOTA", "CHEVROLET", "HONDA", "NISSAN", "BMW", "MERCEDES-BENZ",
        "VOLKSWAGEN", "AUDI", "HYUNDAI", "KIA", "SUBARU", "MAZDA", "JEEP",
        "DODGE", "RAM", "CHRYSLER", "LINCOLN", "CADILLAC", "BUICK", "GMC",
        "LEXUS", "ACURA", "INFINITI", "VOLVO", "PORSCHE", "LAND ROVER",
        "JAGUAR", "MINI", "ALFA ROMEO", "FIAT", "MASERATI", "FERRARI",
        "LAMBORGHINI", "BENTLEY", "ROLLS-ROYCE", "TESLA", "RIVIAN", "LUCID",
        "POLARIS", "CAN-AM", "HARLEY-DAVIDSON", "INDIAN", "KAWASAKI",
        "YAMAHA", "HONDA", "SUZUKI", "DUCATI", "TRIUMPH",
        "FREIGHTLINER", "PETERBILT", "KENWORTH", "MACK", "VOLVO",
        "INTERNATIONAL", "WESTERN STAR", "STERLING",
        "JOHN DEERE", "CATERPILLAR", "CASE",
    ]
    MAX_ADDITIONAL = 500  # Also include 500 more from alphabetical list

    # Build prioritized list: known brands first, then fill with the rest
    all_make_names = {m["Make_Name"].upper(): m for m in all_makes}

    priority_entries = []
    for name in PRIORITY_MAKES:
        if name.upper() in all_make_names:
            priority_entries.append(all_make_names[name.upper()])

    # Add more makes from the full list (skip ones already in priority)
    priority_names_set = {m["Make_Name"].upper() for m in priority_entries}
    additional = [m for m in all_makes if m["Make_Name"].upper() not in priority_names_set]
    additional = additional[:MAX_ADDITIONAL]

    makes_to_use = priority_entries + additional
    total = len(makes_to_use)
    print(f"  Priority makes found: {len(priority_entries)}")
    print(f"  Additional makes: {len(additional)}")

    print(f"\nFetching models for {total} makes (this may take 1-2 minutes)...")
    print("(The NHTSA API has rate limits so we pause briefly between requests)")

    all_text_parts = []

    # Header / intro text — helps the model understand context
    all_text_parts.append(
        "This is a database of vehicle makes and models.\n"
        "A vehicle make is the brand (e.g. Ford, Toyota, BMW).\n"
        "A vehicle model is the specific product (e.g. Mustang, Camry, 3 Series).\n\n"
    )

    success_count = 0
    for i, make_info in enumerate(makes_to_use):

        make_name = make_info["Make_Name"]
        models = fetch_models_for_make(make_name)

        if models:
            text = make_training_text(make_name, models)
            all_text_parts.append(text)
            success_count += 1

        # Progress update every 20 makes
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{total} makes, {success_count} had models...", flush=True)

        # Brief pause to be polite to the API
        time.sleep(0.05)

    full_text = "\n".join(all_text_parts)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print(f"  Total characters: {len(full_text):,}")
    print(f"  Total lines: {full_text.count(chr(10)):,}")
    print(f"  Makes with data: {success_count}")

    # Show a sample so you can inspect the quality
    print("\n--- SAMPLE (first 800 chars) ---")
    print(full_text[:800])
    print("--- END SAMPLE ---")
    print("\nNext step: python 02_tokenizer.py")


if __name__ == "__main__":
    build_dataset()
