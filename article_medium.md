# I'm an Automotive Engineer. I Built My Own AI from Scratch — Here's What I Learned

*No CS degree. No PhD. Just Python, curiosity, and a lot of vehicle data.*

---

I've spent my career working with vehicles — engines, drivetrains, systems engineering. I know how a combustion cycle works, how torque gets transferred through a gearbox, why certain suspension geometries behave the way they do.

But AI? That felt like someone else's territory.

Until a few weeks ago, when I decided to stop treating it like a black box and actually build one myself. Not fine-tune someone else's model, not use an API — build a language model from absolute zero. No shortcuts.

Here's what happened.

---

## Why Build One From Scratch?

There's a difference between *using* a tool and *understanding* it. I've been using AI tools for a while — summarising documents, generating code snippets, explaining technical specs. But I had no real idea what was happening under the hood.

As an engineer, that bothered me.

So I set a goal: build a working language model that knows about vehicles, train it on real data, and then fine-tune it to become an Audi expert. The whole thing — architecture, training, data pipeline — written in Python from scratch.

---

## What Even Is a Language Model?

Before I get into the build, let me explain what these things actually do — because once I understood it, the whole field made a lot more sense.

A language model doesn't "understand" text the way you and I do. It's a statistical pattern-matching machine. Given a sequence of text, it learns to predict: *what character (or word) comes next?*

That's it. That's the core of GPT-4, Claude, Gemini — all of them. Predict the next token. Do it billions of times on trillions of examples. Out comes a system that appears to reason, write, and answer questions.

The magic isn't in some mysterious intelligence. It's in the fact that to consistently predict the next word, you *have to* learn grammar, facts, reasoning, context — everything. Prediction forces understanding.

---

## The Architecture: A Tiny GPT

I built what's called a **transformer** — the same architecture used by every major language model today. The key innovation inside a transformer is called **self-attention**.

Here's the intuition. When you read the sentence *"The Ford Mustang is a sports car"*, your brain connects *Mustang* to *Ford* and *sports car* automatically. Self-attention is the mechanism that teaches the model to make those connections mathematically.

Every token (in my case, every character) creates three vectors:
- **Query**: what am I looking for?
- **Key**: what do I offer?
- **Value**: what information do I carry?

The model computes similarity scores between every query and every key, turns those into weights, and produces a weighted blend of all the values. That weighted blend is the attention output — each character now "knows" which other characters are most relevant to it.

Stack four of these attention layers with some feed-forward networks between them, add 218,000 learnable parameters, and you have my model. Tiny compared to the industry (GPT-4 has roughly 1.8 *trillion* parameters), but structurally identical.

---

## The Data: Where It Gets Interesting

I used the **NHTSA's free public API** — the US vehicle registration database. It has every make and model ever officially registered in the United States. Over 12,000 makes.

I fetched models for the top 1,000 makes and converted the structured data into natural language training text. The key insight here: the format of your data shapes what the model learns. If you feed it raw CSV:

```
Ford,Mustang,1964
```

It learns to complete CSV rows. Instead, I formatted it as Q&A pairs:

```
Q: What models does Ford make?
A: Ford makes the Mustang, F-150, Explorer...

Q: Who makes the Mustang?
A: The Mustang is made by Ford.
```

Now the model learns to answer questions. Same data, completely different outcome.

After 3,000 training steps — about 25 minutes on my laptop CPU — the loss dropped from 4.41 (random guessing) to 1.20. The model could generate grammatically sensible vehicle Q&A text. Not perfect, but real.

---

## The Fine-Tuning: Going Audi-Only

Once the base model was trained, I fine-tuned it specifically on Audi.

Fine-tuning is how every production AI model in the world works. You pre-train on enormous amounts of general data, then train further on specific data for your use case. GPT-4 was pre-trained on basically the entire internet, then fine-tuned to follow instructions and be helpful.

My version was humbler: train on 450 vehicle makes, then specialise on one.

For the Audi dataset, I used three sources:
1. **Wikipedia** — full articles on Audi, the A4, A6, A8, Q5, Q7, R8, TT, e-tron, and the quattro system. That's about 320,000 characters of real encyclopaedic content.
2. **NHTSA** — all 55 official Audi models registered in the US.
3. **Handcrafted Q&A** — I manually wrote precise answers to the questions I wanted the model to handle: what RS stands for (Rennsport — German for race sport), the history of quattro, the model lineup hierarchy, when each model was introduced.

The critical difference in fine-tuning: I used a **learning rate 6 times lower** than pre-training. This is the most important hyperparameter change. If you use a large learning rate on a small new dataset, the model takes huge steps and overwrites everything it previously learned — this is called *catastrophic forgetting*. A small learning rate makes careful adjustments that layer new knowledge on top of existing knowledge.

---

## What Worked, What Didn't, and What I Learned

The results were honest.

The fine-tuned model clearly learned the Audi vocabulary — every response mentioned A4, A6, A7, A8, RS, quattro, Sportback. The base model, asked the same questions, would respond with completely invented makes like "Afturindang" or "Adironda." The improvement is real.

But — and this is the most important thing I learned — **the model is too small to reliably retrieve specific facts**.

Ask it "What does RS stand for?" and instead of saying "Rennsport", it generates something in the right neighbourhood but gets confused. This isn't a bug in my code. It's a fundamental limitation of scale.

To reliably *store* and *recall* factual associations, a model needs enough capacity in its weights. My 218,000 parameters can learn patterns and vocabulary. But GPT-2 has 117 million — 537 times bigger — and only at that scale do precise factual answers start to become reliable.

This is why the AI industry obsesses over scale. Bigger isn't just better marketing — it's a real technical threshold between "learned the vibe" and "learned the facts."

---

## What This Project Actually Gave Me

I went in wanting to demystify AI. I came out with something more useful: a *mechanical* understanding of these systems.

I know how attention works because I implemented it. I know why learning rate matters because I watched catastrophic forgetting happen. I know why "more data" isn't always the answer because I watched my model saturate at 1.20 loss with the data I had.

More practically: I now look at tools like ChatGPT or Copilot and understand — at least structurally — what's happening inside them. When someone talks about fine-tuning, context windows, or hallucinations, I have a physical intuition for what those words actually mean.

For an engineer, that's the real value. Not that I built GPT-4. But that I'm no longer intimidated by the people who did.

---

## What's Next

The logical next step is to take this exact fine-tuning pipeline and apply it to an actual pre-trained model — something like GPT-2 from Hugging Face. Same code I already wrote, 537 times more parameters, dramatically better Audi answers. The architecture, data pipeline, and training loop are all in place. Only the model size changes.

If you're an engineer curious about AI and you haven't tried building something from scratch — do it. You don't need a machine learning background. You need Python, a few weeks, and the willingness to watch loss curves for longer than is strictly comfortable.

The code for this project is on GitHub. The model runs on a CPU. The data is free.

Go build something.

---

*Built with PyTorch, NHTSA public API, Wikipedia, and way too much coffee.*
*Questions? Connect with me on LinkedIn.*
