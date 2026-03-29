"""
STEP 7: Build the Audi Fine-Tuning Dataset
==========================================
This script builds a rich, Audi-focused training dataset from THREE sources:

  1. Wikipedia  — encyclopaedic prose: history, specs, model details
  2. NHTSA API  — official registered model names → Q&A pairs
  3. Handcrafted Q&A — key Audi facts written in many phrasings

KEY CONCEPT: Why Do We Need Multiple Data Sources?
===================================================
Each source contributes something different:

  Wikipedia:   DEPTH — real facts, years, engine types, history
               "The Audi A4 is a line of compact executive cars produced
                since November 1994 by the German car manufacturer Audi."
               The model learns TRUE facts in natural language.

  NHTSA:       BREADTH — every official model name, in Q&A format
               Ensures the model knows the full lineup, not just famous models.

  Handcrafted: PRECISION — the exact Q&A patterns we want the model to produce
               "Q: What does RS stand for?\nA: RS stands for Rennsport..."
               We write the ideal answers ourselves.

KEY CONCEPT: Data Cleaning
===========================
Wikipedia text has noise: section headers (== History ==), multiple blank lines,
citation markers, table artifacts. We clean this out so the model sees
clean, natural prose — not formatting artifacts.

KEY CONCEPT: The 20% General Mix
==================================
We include 20% of the original vehicles.txt in this fine-tuning set.
This PREVENTS CATASTROPHIC FORGETTING — the phenomenon where training
on new data completely overwrites previously learned knowledge.
By keeping some general vehicle text, the model stays grounded.

Run: python3.12 07_audi_data.py
"""

import requests
import os
import re
import time
import random

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "audi_finetune.txt")
GENERAL_FILE = os.path.join(DATA_DIR, "vehicles.txt")

WIKI_API = "https://en.wikipedia.org/w/api.php"
NHTSA_API = "https://vpic.nhtsa.dot.gov/api"
HEADERS = {"User-Agent": "MyLLM-AudiProject/1.0 (educational)"}

# Wikipedia pages to fetch (main Audi article + key model pages)
WIKI_PAGES = [
    "Audi",
    "Audi A3",
    "Audi A4",
    "Audi A6",
    "Audi A8",
    "Audi Q5",
    "Audi Q7",
    "Audi R8",
    "Audi TT",
    "Audi e-tron",
    "Audi quattro",
    "Audi RS models",
]


# ════════════════════════════════════════════════════════════════════
# PART A: Wikipedia
# ════════════════════════════════════════════════════════════════════

def fetch_wikipedia_page(title):
    """Fetch plain text of a Wikipedia article via the MediaWiki API."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        page = list(pages.values())[0]
        return page.get("extract", "")
    except Exception as e:
        print(f"  Warning: could not fetch '{title}': {e}")
        return ""


def clean_wikipedia_text(text):
    """
    Clean Wikipedia plain text for use as training data.

    Removes:
    - Section headers (== History ==, === Production ===)
    - Multiple consecutive blank lines
    - Lines that are just references or short navigation artifacts
    - Citation-style text

    Keeps:
    - All natural prose paragraphs
    - Numbers, years, model names, technical specs
    """
    # Remove section headers like "== History ==" or "=== Models ==="
    text = re.sub(r"={2,}[^=\n]+={2,}", "", text)

    # Remove lines that are just whitespace
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # Skip very short lines (likely navigation artifacts) but keep content
        if len(line) > 5:
            cleaned.append(line)

    # Rejoin with single newlines
    text = "\n".join(cleaned)

    # Collapse 3+ consecutive newlines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove parenthetical references like "(pp. 45–67)" or "[citation needed]"
    text = re.sub(r"\[.*?\]", "", text)

    return text.strip()


def fetch_all_wikipedia():
    """Fetch and clean all Wikipedia pages, return combined text."""
    parts = []
    total_chars = 0
    for title in WIKI_PAGES:
        print(f"  Fetching Wikipedia: '{title}'...", flush=True)
        raw = fetch_wikipedia_page(title)
        if raw:
            cleaned = clean_wikipedia_text(raw)
            parts.append(f"--- {title} ---\n{cleaned}\n")
            total_chars += len(cleaned)
            print(f"    → {len(cleaned):,} chars", flush=True)
        time.sleep(0.3)  # Be polite to Wikipedia's servers
    print(f"  Total Wikipedia text: {total_chars:,} chars")
    return "\n\n".join(parts)


# ════════════════════════════════════════════════════════════════════
# PART B: NHTSA API (Audi model names → Q&A)
# ════════════════════════════════════════════════════════════════════

def fetch_audi_models_nhtsa():
    """Fetch all Audi models from NHTSA and convert to Q&A pairs."""
    print("  Fetching Audi models from NHTSA API...", flush=True)
    try:
        url = f"{NHTSA_API}/vehicles/getmodelsformake/audi?format=json"
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        models = sorted(set(x["Model_Name"] for x in r.json()["Results"] if x.get("Model_Name")))
        print(f"  Found {len(models)} Audi models: {models[:8]}...", flush=True)
    except Exception as e:
        print(f"  Warning: NHTSA fetch failed ({e}), using fallback list", flush=True)
        # Fallback: hardcoded list from earlier successful fetch
        models = [
            "A1", "A3", "A4", "A5", "A6", "A7", "A8",
            "Q2", "Q3", "Q4 e-tron", "Q5", "Q7", "Q8",
            "S3", "S4", "S5", "S6", "S7", "S8",
            "RS 3", "RS 4", "RS 5", "RS 6", "RS 7", "RS Q3", "RS Q8",
            "TT", "TTS", "TT RS", "R8", "allroad", "e-tron", "e-tron GT",
        ]

    model_list = ", ".join(models)
    short_list = ", ".join(models[:12])

    lines = []
    # Declarative
    lines.append(f"Audi makes the following models: {model_list}.\n")

    # Standard Q&A
    lines.append(f"Q: What models does Audi make?\nA: Audi makes many models including the {short_list}, and more.\n")
    lines.append(f"Q: What vehicles does Audi produce?\nA: Audi produces the {short_list}, among others.\n")
    lines.append(f"Q: Can you list some Audi models?\nA: Some Audi models include the {short_list}.\n")

    # Per-model membership Q&A
    for model in models:
        lines.append(f"Q: Is the {model} an Audi model?\nA: Yes, the {model} is a model made by Audi.\n")
        lines.append(f"Q: Who makes the Audi {model}?\nA: The Audi {model} is made by Audi AG.\n")

    return "\n".join(lines) + "\n"


# ════════════════════════════════════════════════════════════════════
# PART C: Handcrafted Q&A Knowledge Base
# ════════════════════════════════════════════════════════════════════

AUDI_QA = """
Audi AG is a German manufacturer of luxury and performance vehicles headquartered in Ingolstadt, Bavaria, Germany.
Audi is a wholly owned subsidiary of the Volkswagen Group.
Audi was founded in 1909 by August Horch.
The four rings in the Audi logo represent the four companies that merged to form Auto Union in 1932: Audi, DKW, Horch, and Wanderer.

Q: Where is Audi headquartered?
A: Audi is headquartered in Ingolstadt, Bavaria, Germany.

Q: Who founded Audi?
A: Audi was founded by August Horch in 1909.

Q: What does the Audi logo mean?
A: The four rings in the Audi logo represent the four companies that merged to form Auto Union in 1932: Audi, DKW, Horch, and Wanderer.

Q: What group does Audi belong to?
A: Audi is a wholly owned subsidiary of the Volkswagen Group.

Q: What does Audi mean?
A: Audi is the Latin translation of the German surname Horch, meaning to hark or listen.

--- Audi Model Lines ---

Audi organises its vehicles into several distinct model lines.
The A-series consists of Audi's standard luxury sedans and wagons: the A1, A3, A4, A5, A6, A7, and A8.
The Q-series consists of Audi's SUVs and crossovers: the Q2, Q3, Q4, Q5, Q7, and Q8.
The S-series are sportier, higher performance versions of the A-series models: S3, S4, S5, S6, S7, and S8.
The RS series are Audi's highest performance models. RS stands for Rennsport, which is German for race sport.
The TT is a sports coupe available in standard, TTS, and TT RS variants.
The R8 is Audi's mid-engine supercar featuring a naturally aspirated V10 engine.
The e-tron lineup consists of Audi's fully electric vehicles.

Q: What is the Audi A-series?
A: The Audi A-series is Audi's standard luxury lineup of sedans and wagons. It includes the A1, A3, A4, A5, A6, A7, and A8, ranging from compact to full-size luxury.

Q: What is the Audi Q-series?
A: The Audi Q-series is Audi's lineup of SUVs and crossovers. It includes the Q2, Q3, Q4, Q5, Q7, and Q8.

Q: What is the Audi S-series?
A: The Audi S-series are sportier, higher-performance versions of the A-series models. For example, the S4 is a performance version of the A4.

Q: What does RS stand for in Audi model names?
A: RS stands for Rennsport, which is German for race sport. Audi RS models are the highest performance variants of each model line.

Q: What is an Audi RS model?
A: Audi RS models are the highest performance variants in Audi's lineup. RS stands for Rennsport, meaning race sport in German. Examples include the RS 4, RS 6, RS Q8, and RS e-tron GT.

Q: What is the difference between the Audi A4 and S4?
A: The Audi S4 is a high-performance version of the A4. The S4 has a more powerful engine, sportier suspension, and sport-oriented interior compared to the standard A4.

Q: What is the difference between the Audi S4 and RS 4?
A: The RS 4 is more powerful and track-focused than the S4. The RS 4 is Audi's top performance version of the A4, with even more power and sportier tuning than the S4.

Q: What is the performance hierarchy in Audi models?
A: The performance hierarchy in Audi models goes from standard to sport to race: A-series (standard) then S-series (sport) then RS-series (race sport, highest performance).

--- Audi quattro ---

Audi quattro is Audi's permanent all-wheel-drive system.
Audi quattro was first introduced in 1980 on the original Audi Quattro coupe.
The Audi quattro system distributes power to all four wheels, giving better traction and handling.
Quattro is available across most Audi model lines.
The original Audi Quattro rally car dominated the World Rally Championship in the early 1980s.

Q: What is Audi quattro?
A: Audi quattro is Audi's legendary permanent all-wheel-drive system, first introduced in 1980. It distributes power to all four wheels for better traction and handling in all conditions.

Q: When was Audi quattro introduced?
A: Audi quattro all-wheel drive was first introduced in 1980 on the original Audi Quattro coupe.

Q: Is Audi quattro available on all models?
A: Quattro all-wheel drive is available on most Audi models and is standard on all S and RS variants.

--- Audi A4 ---

The Audi A4 is a compact executive car that has been produced since 1994.
The A4 is one of Audi's best-selling models worldwide.
The A4 is available as a sedan and an Avant wagon.
The Audi S4 is the high-performance variant of the A4.
The Audi RS 4 is the top performance variant of the A4.

Q: What is the Audi A4?
A: The Audi A4 is a compact executive car that has been in production since 1994. It is one of Audi's best-selling models and is available as a sedan and an Avant wagon.

Q: When was the Audi A4 first produced?
A: The Audi A4 has been in production since 1994. It replaced the Audi 80 in Audi's lineup.

Q: What body styles does the Audi A4 come in?
A: The Audi A4 is available as a sedan and as an Avant, which is a station wagon.

--- Audi Q7 ---

The Audi Q7 is a full-size luxury SUV produced by Audi.
The Q7 was first introduced in 2005 as Audi's first SUV.
The Q7 is a three-row SUV, meaning it has seven seats.
The Audi Q8 is a sportier, coupe-styled version of the Q7 platform.
The SQ7 is the high-performance variant of the Q7.

Q: What is the Audi Q7?
A: The Audi Q7 is a full-size three-row luxury SUV. It was first introduced in 2005 and was Audi's first SUV. It seats up to seven passengers.

Q: When was the Audi Q7 first produced?
A: The Audi Q7 was first introduced in 2005. It was Audi's first SUV model.

Q: How many seats does the Audi Q7 have?
A: The Audi Q7 is a three-row SUV that seats up to seven passengers.

Q: What is the difference between the Q7 and Q8?
A: The Audi Q8 is a sportier, coupe-styled SUV based on the Q7 platform. The Q7 has a more traditional SUV shape with three rows of seats, while the Q8 has a sloping roofline and focuses on two rows.

--- Audi R8 ---

The Audi R8 is Audi's mid-engine supercar.
The R8 was first introduced in 2006.
The Audi R8 features a naturally aspirated V10 engine in its most powerful form.
The R8 shares its platform with the Lamborghini Huracan.
The R8 is built at Audi's Neckarsulm facility.

Q: What is the Audi R8?
A: The Audi R8 is Audi's mid-engine supercar. Introduced in 2006, it features a naturally aspirated V10 engine in its top variant and shares its platform with the Lamborghini Huracan.

Q: When was the Audi R8 introduced?
A: The Audi R8 was first introduced in 2006.

Q: What engine does the Audi R8 have?
A: The Audi R8 is available with a V10 naturally aspirated engine in its most powerful form. It is a mid-engine supercar.

Q: What car does the Audi R8 share its platform with?
A: The Audi R8 shares its platform with the Lamborghini Huracan.

Q: What is Audi's most powerful production car?
A: The Audi R8 with the V10 engine is one of Audi's most powerful production cars. The RS e-tron GT is also among Audi's most powerful vehicles.

--- Audi TT ---

The Audi TT is a sports coupe that has been in production since 1998.
The TT is available as a coupe and a roadster convertible.
The TTS is a higher performance version of the TT.
The TT RS is the top performance variant of the TT, with a five-cylinder turbocharged engine.

Q: What is the Audi TT?
A: The Audi TT is a compact sports coupe that has been in production since 1998. It is available as a coupe and as a roadster convertible.

Q: When was the Audi TT first made?
A: The Audi TT has been in production since 1998.

Q: What is the TT RS?
A: The Audi TT RS is the top performance variant of the TT lineup. It features a five-cylinder turbocharged engine and is the most powerful version of the TT.

--- Audi Electric Vehicles ---

Audi's electric vehicle lineup is called the e-tron.
The Audi e-tron was Audi's first fully electric SUV.
The Audi e-tron GT is a fully electric gran turismo sports car.
The e-tron GT shares its platform with the Porsche Taycan.
The Audi Q4 e-tron is a compact electric SUV.
The Audi Q8 e-tron is Audi's full-size electric SUV.

Q: What are Audi's electric vehicles?
A: Audi's electric vehicle lineup is branded as e-tron. It includes the e-tron SUV, e-tron GT sports car, Q4 e-tron compact SUV, and Q8 e-tron full-size SUV.

Q: What is the Audi e-tron?
A: The Audi e-tron is Audi's fully electric vehicle range. The original e-tron was Audi's first fully electric SUV. The e-tron GT is a fully electric gran turismo sports car that shares its platform with the Porsche Taycan.

Q: What is the Audi e-tron GT?
A: The Audi e-tron GT is a fully electric gran turismo sports car. It shares its platform with the Porsche Taycan and is available in a standard and high-performance RS e-tron GT variant.

Q: What platform does the Audi e-tron GT share?
A: The Audi e-tron GT shares its platform with the Porsche Taycan.

--- Flagship Models ---

The Audi A8 is Audi's flagship full-size luxury sedan.
The Audi Q8 is Audi's flagship luxury SUV.
The Audi R8 is Audi's flagship supercar.
The Audi A8 competes with vehicles like the BMW 7 Series and Mercedes-Benz S-Class.

Q: What is Audi's flagship sedan?
A: The Audi A8 is Audi's flagship full-size luxury sedan, positioned at the top of the A-series lineup.

Q: What is Audi's flagship SUV?
A: The Audi Q8 is Audi's flagship luxury SUV.

Q: What is Audi's flagship car?
A: Audi's flagship lineup includes the A8 (flagship sedan), Q8 (flagship SUV), and R8 (flagship supercar).

--- Audi allroad ---

The Audi allroad is a raised, all-terrain version of the Audi A4 Avant or A6 Avant wagon.
The allroad has higher ground clearance than standard Audi wagons.
The allroad is designed for light off-road use and rough terrain.

Q: What is the Audi allroad?
A: The Audi allroad is a raised, all-terrain version of the Audi Avant wagon. It has higher ground clearance and is designed for light off-road use.

--- General Audi Facts ---

Audi vehicles are known for high build quality, advanced technology, and performance.
Audi's Virtual Cockpit is a fully digital instrument cluster replacing traditional gauges.
Audi's MMI system is its multimedia interface for infotainment control.
Audi Sport GmbH is the subsidiary responsible for RS and R8 models.

Q: What is Audi known for?
A: Audi is known for luxury vehicles, advanced technology, and high-performance cars. Audi is particularly known for its quattro all-wheel-drive system and its RS performance models.

Q: What is Audi Sport GmbH?
A: Audi Sport GmbH is the subsidiary of Audi responsible for developing and producing the RS and R8 high-performance models.

Q: What country is Audi from?
A: Audi is a German company, headquartered in Ingolstadt, Bavaria, Germany.

Q: Is Audi a luxury brand?
A: Yes, Audi is a German luxury automotive brand. It produces luxury vehicles ranging from compact cars to full-size sedans, SUVs, and supercars.
"""


# ════════════════════════════════════════════════════════════════════
# PART D: Mix in general vehicle data (prevents catastrophic forgetting)
# ════════════════════════════════════════════════════════════════════

def sample_general_data(fraction=0.20):
    """Sample a fraction of the original vehicles.txt as a regulariser."""
    if not os.path.exists(GENERAL_FILE):
        return ""
    with open(GENERAL_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Randomly sample lines
    random.seed(42)
    n = int(len(lines) * fraction)
    sampled = random.sample(lines, n)
    return "".join(sampled)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def build_audi_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 55)
    print("Building Audi Fine-Tuning Dataset")
    print("=" * 55)

    parts = []

    # A: Wikipedia
    print("\n[Part A] Fetching Wikipedia articles...")
    wiki_text = fetch_all_wikipedia()
    parts.append("=== WIKIPEDIA CONTENT ===\n" + wiki_text)

    # B: NHTSA
    print("\n[Part B] Fetching NHTSA Audi model data...")
    nhtsa_text = fetch_audi_models_nhtsa()
    parts.append("=== NHTSA MODEL DATA ===\n" + nhtsa_text)

    # C: Handcrafted Q&A
    print("\n[Part C] Adding handcrafted Audi Q&A knowledge base...")
    parts.append("=== AUDI KNOWLEDGE BASE ===\n" + AUDI_QA.strip())
    print(f"  Handcrafted Q&A: {len(AUDI_QA):,} chars", flush=True)

    # D: General vehicles mix (catastrophic forgetting prevention)
    print("\n[Part D] Sampling 20% of general vehicles.txt for stability...")
    general_sample = sample_general_data(fraction=0.20)
    parts.append("=== GENERAL VEHICLE MIX ===\n" + general_sample)
    print(f"  General sample: {len(general_sample):,} chars", flush=True)

    # Combine and save
    full_text = "\n\n".join(parts)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\n{'='*55}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total characters: {len(full_text):,}")
    print(f"Total lines:      {full_text.count(chr(10)):,}")

    # Spot-check key terms
    key_terms = ["quattro", "Rennsport", "Ingolstadt", "R8", "e-tron", "RS", "A4", "Q7"]
    print("\nKey term check:")
    for term in key_terms:
        count = full_text.count(term)
        status = "✓" if count > 0 else "✗ MISSING"
        print(f"  '{term}': {count} occurrences {status}")

    print("\n--- SAMPLE (first 600 chars) ---")
    print(full_text[:600])
    print("\nNext step: python3.12 08_finetune.py")


if __name__ == "__main__":
    build_audi_dataset()
