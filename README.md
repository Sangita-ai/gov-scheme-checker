# 🇮🇳 Government Scheme Eligibility Checker

Ever wondered which government schemes you actually qualify for? Most people miss out on benefits they deserve simply because they don't know about them or find the process too confusing.

This tool fixes that. Answer a few simple questions in a chat — in English, Hindi, or Bengali — and within minutes you'll know exactly which of 25+ central and West Bengal state schemes you're eligible for, along with a step-by-step guide to apply.

**Live demo:** [https://sangita-ai.github.io/gov-scheme-checker/]

---

## What it does

You chat with an AI advisor (like WhatsApp, but for government schemes). It asks you things like your age, occupation, income, and state. Based on your answers it instantly checks schemes like PM-KISAN, Ayushman Bharat, Lakshmir Bhandar, PM Mudra Loan, NSP Scholarships, and 20+ more.

For every scheme you qualify for, you get:
- Why you're eligible (plain language, not legal jargon)
- Exactly which documents to gather
- Where to apply and the official link
- The helpline number to call if you get stuck

No login. No data stored. Completely free.

---

## Schemes covered

**Central Government (20 schemes)**
PM-KISAN · Ayushman Bharat · PM Awas Yojana · PM Ujjwala · PM Mudra (Shishu & Kishore) · PM SVANidhi · Sukanya Samridhi · Jan Dhan · PMSBY · PMJJBY · Atal Pension · PM Fasal Bima · MGNREGS · NSP Scholarship · PM Vishwakarma · PMEGP · Stand Up India · Kisan Credit Card · PM Scholarship (PMSS)

**West Bengal State (5 schemes)**
Lakshmir Bhandar · Kanyashree K2 · Krishak Bandhu · Swasthya Sathi · Rupashree

---

## Tech stack

| What | Why |
|------|-----|
| FastAPI + Python | Lightweight backend, runs comfortably on 8GB RAM |
| SQLite | Zero-setup database, no separate server needed |
| Groq API (free) | Super fast AI responses using llama-3.1-8b-instant |
| Gemini API (free) | Fallback if Groq hits rate limits, handles Hindi/Bengali natively |
| Vanilla HTML/CSS/JS | No build step, no npm, opens in any browser |
| Web Speech API | Voice input built into Chrome, no extra setup |

No Docker. No PostgreSQL. No Node.js. Intentionally lightweight so it runs on a basic laptop.

---


## Setup (Windows)

```bash

git clone https://github.com/YOUR_USERNAME/gov-scheme-checker.git
cd gov-scheme-checker

cd backend
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

copy .env.example .env

python seed_db.py

uvicorn main:app --reload --port 8000
```

Then in a second terminal:
```bash
cd frontend
python -m http.server 8080 --bind 0.0.0.0
```

Open `http://localhost:8080` in your browser.

---

## How the AI works

The interesting part: instead of blindly trusting the LLM to extract answers, the app uses a two-layer approach.

**Layer 1 — Python rule engine** handles yes/no answers, numbers, state names, occupations using regex. This is 100% reliable. "Yes", "haan", "হ্যাঁ" all work correctly and get saved to the right field immediately.

**Layer 2 — LLM (Groq/Gemini)** only generates the friendly reply text and next question. It never touches the data. This is what prevents the classic bug where a chatbot asks you the same question 6 times.

Eligibility checking itself is also pure Python rule logic — no LLM involved. Fast, predictable, and accurate.

---