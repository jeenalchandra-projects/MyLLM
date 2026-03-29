"""
Audi Vehicle LLM — Web Inference App
=====================================
Flask app serving the fine-tuned Audi language model.
Deployable to Railway, Render, or any Python hosting.

Local dev:  python app.py
Production: gunicorn app:app --bind 0.0.0.0:$PORT
"""

import os
import sys
import json
import torch

from flask import Flask, request, jsonify, render_template_string

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from model_def import MiniGPT, BLOCK_SIZE

app = Flask(__name__)

# ── Load model at startup ─────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
MODEL_PATH = os.path.join(MODEL_DIR, "audi_model.pt")

print("Loading vocabulary...")
with open(VOCAB_PATH) as f:
    char_to_int = json.load(f)
int_to_char = {int(v): k for k, v in char_to_int.items()}
vocab_size = len(char_to_int)

print("Loading model...")
ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = MiniGPT(vocab_size)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model ready — {sum(p.numel() for p in model.parameters()):,} parameters")


def generate(prompt: str, max_new: int = 200, temperature: float = 0.7) -> str:
    """Run inference: prompt → generated continuation."""
    clean = "".join(c for c in prompt if c in char_to_int)
    ids   = [char_to_int[c] for c in clean]
    idx   = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, max_new, temperature=temperature)
    full = "".join(int_to_char[i] for i in out[0].tolist())
    return full[len(clean):]


# ── HTML frontend (embedded) ─────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Audi LLM — Vehicle Intelligence</title>
  <style>
    :root {
      --red:    #BB0A21;
      --dark:   #0D0D0D;
      --card:   #161616;
      --border: #2A2A2A;
      --text:   #E8E8E8;
      --muted:  #888;
      --accent: #C0C0C0;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--dark);
      color: var(--text);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem 4rem;
    }

    /* Header */
    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    .rings {
      display: flex;
      justify-content: center;
      gap: -8px;
      margin-bottom: 1.2rem;
    }
    .ring {
      width: 36px; height: 36px;
      border-radius: 50%;
      border: 3px solid var(--accent);
      margin-left: -10px;
      background: transparent;
    }
    .ring:first-child { margin-left: 0; }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      color: #fff;
    }
    h1 span { color: var(--red); }
    .subtitle {
      font-size: 0.9rem;
      color: var(--muted);
      margin-top: 0.4rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    /* Card */
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2rem;
      width: 100%;
      max-width: 700px;
    }

    label {
      display: block;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.5rem;
    }

    textarea, .output-box {
      width: 100%;
      background: #0A0A0A;
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 0.9rem;
      padding: 0.9rem 1rem;
      line-height: 1.6;
      resize: vertical;
    }
    textarea { min-height: 90px; }
    textarea:focus {
      outline: none;
      border-color: var(--red);
    }

    /* Controls row */
    .controls {
      display: flex;
      align-items: center;
      gap: 1rem;
      margin: 1rem 0;
      flex-wrap: wrap;
    }
    .slider-group {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      flex: 1;
    }
    .slider-group label {
      margin: 0;
      white-space: nowrap;
      font-size: 0.7rem;
    }
    input[type=range] {
      flex: 1;
      accent-color: var(--red);
    }
    #temp-val, #len-val {
      font-size: 0.8rem;
      color: var(--accent);
      min-width: 2rem;
      text-align: right;
    }

    /* Preset chips */
    .presets {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 1.2rem;
    }
    .chip {
      background: #1E1E1E;
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 0.3rem 0.8rem;
      font-size: 0.75rem;
      color: var(--accent);
      cursor: pointer;
      transition: all 0.15s;
      white-space: nowrap;
    }
    .chip:hover {
      border-color: var(--red);
      color: #fff;
    }

    /* Generate button */
    button#gen-btn {
      background: var(--red);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 0.75rem 2rem;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
      letter-spacing: 0.04em;
      white-space: nowrap;
    }
    button#gen-btn:hover  { opacity: 0.88; }
    button#gen-btn:active { transform: scale(0.97); }
    button#gen-btn:disabled { opacity: 0.4; cursor: not-allowed; }

    /* Output */
    .output-section { margin-top: 1.5rem; }
    .output-box {
      min-height: 120px;
      color: #A8D8A8;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .output-box.loading { color: var(--muted); font-style: italic; }

    /* Divider */
    .divider {
      border: none;
      border-top: 1px solid var(--border);
      margin: 1.5rem 0;
    }

    /* Footer */
    footer {
      margin-top: 2.5rem;
      text-align: center;
      font-size: 0.75rem;
      color: var(--muted);
      line-height: 1.8;
    }
    footer a { color: var(--accent); text-decoration: none; }

    /* Pulse animation for loading */
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    .pulsing { animation: pulse 1s infinite; }
  </style>
</head>
<body>

  <header>
    <div class="rings">
      <div class="ring"></div>
      <div class="ring"></div>
      <div class="ring"></div>
      <div class="ring"></div>
    </div>
    <h1>Audi <span>LLM</span></h1>
    <p class="subtitle">Vehicle Intelligence · Built from Scratch · PyTorch</p>
  </header>

  <div class="card">
    <label>Prompt</label>
    <textarea id="prompt" placeholder="Q: What models does Audi make?&#10;A:">Q: What models does Audi make?
A:</textarea>

    <p style="font-size:0.72rem;color:var(--muted);margin:0.4rem 0 1rem;">
      Format your prompt as <code style="color:var(--accent)">Q: your question?&nbsp;&nbsp;A:</code>
    </p>

    <label>Quick Questions</label>
    <div class="presets">
      <span class="chip" onclick="setPrompt('Q: What does RS stand for in Audi models?\\nA:')">What does RS mean?</span>
      <span class="chip" onclick="setPrompt('Q: What is Audi quattro?\\nA:')">What is quattro?</span>
      <span class="chip" onclick="setPrompt('Q: What is the Audi R8?\\nA:')">Tell me about the R8</span>
      <span class="chip" onclick="setPrompt('Q: What are Audi\\'s electric vehicles?\\nA:')">Electric vehicles</span>
      <span class="chip" onclick="setPrompt('Q: What is Audi\\'s flagship sedan?\\nA:')">Flagship sedan</span>
      <span class="chip" onclick="setPrompt('Q: Where is Audi headquartered?\\nA:')">Where is Audi from?</span>
    </div>

    <hr class="divider">

    <div class="controls">
      <div class="slider-group">
        <label>Temperature</label>
        <input type="range" id="temp" min="0.2" max="1.5" step="0.1" value="0.7"
               oninput="document.getElementById('temp-val').textContent=this.value">
        <span id="temp-val">0.7</span>
      </div>
      <div class="slider-group">
        <label>Length</label>
        <input type="range" id="length" min="50" max="300" step="10" value="150"
               oninput="document.getElementById('len-val').textContent=this.value">
        <span id="len-val">150</span>
      </div>
      <button id="gen-btn" onclick="generate()">Generate ▶</button>
    </div>

    <div class="output-section">
      <label>Response</label>
      <div class="output-box" id="output">Your answer will appear here...</div>
    </div>
  </div>

  <footer>
    Built with PyTorch · Trained on NHTSA + Wikipedia · 218K parameters<br>
    <a href="https://medium.com" target="_blank">Read the full story on Medium</a>
  </footer>

  <script>
    function setPrompt(text) {
      document.getElementById('prompt').value = text.replace(/\\n/g, '\n');
    }

    async function generate() {
      const prompt = document.getElementById('prompt').value.trim();
      const temp   = parseFloat(document.getElementById('temp').value);
      const length = parseInt(document.getElementById('length').value);
      const btn    = document.getElementById('gen-btn');
      const out    = document.getElementById('output');

      if (!prompt) return;

      btn.disabled = true;
      btn.textContent = 'Generating...';
      out.textContent = 'Thinking...';
      out.classList.add('loading', 'pulsing');

      try {
        const res = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, temperature: temp, max_new: length })
        });
        const data = await res.json();
        out.classList.remove('loading', 'pulsing');
        if (data.error) {
          out.textContent = 'Error: ' + data.error;
        } else {
          // Show prompt + response together
          out.textContent = prompt + data.generated;
        }
      } catch (e) {
        out.classList.remove('loading', 'pulsing');
        out.textContent = 'Request failed: ' + e.message;
      }

      btn.disabled = false;
      btn.textContent = 'Generate ▶';
    }

    // Allow Ctrl+Enter to generate
    document.getElementById('prompt').addEventListener('keydown', e => {
      if (e.ctrlKey && e.key === 'Enter') generate();
    });
  </script>

</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data        = request.get_json(force=True)
    prompt      = data.get("prompt", "").strip()
    temperature = float(data.get("temperature", 0.7))
    max_new     = int(data.get("max_new", 150))

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Clamp values to safe ranges
    temperature = max(0.1, min(2.0, temperature))
    max_new     = max(20,  min(400, max_new))

    try:
        result = generate(prompt, max_new=max_new, temperature=temperature)
        return jsonify({"generated": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "audi_model.pt", "params": 217937})


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
