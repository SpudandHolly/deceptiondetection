# Deception Detection Lab

**Forensic Statement & Linguistic Analysis (FSLA) Tool**

A web-based application that analyzes text for linguistic deception indicators, based on proven FSLA methodology used by investigators, insurers, and analysts.

![Risk Levels](https://img.shields.io/badge/Risk%20Levels-Low%20|%20Medium%20|%20High%20|%20Very%20High-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-2.0+-orange)

## Features

- **30+ Deception Indicators** across multiple categories
- **Real-time Analysis** with instant results
- **Color-coded Highlighting** of suspicious phrases
- **Risk Scoring** (0-100 scale)
- **Actionable Recommendations** based on findings
- **Insurance Fraud Detection** specialized indicators

## Indicator Categories

| Category | What It Detects |
|----------|-----------------|
| **Pronouns** | Distancing language, pronoun shifts |
| **Temporal** | Vague time references, missing periods |
| **Hedging** | Qualifiers, uncertainty language |
| **Denials** | Weak denials, non-denial denials |
| **Emotional** | Bolstering, misplaced emotion |
| **Commitment** | Passive voice, conditional language |
| **Corporate** | Double-speak, earnings call red flags |
| **Manipulation** | Love bombing, urgency creation |
| **Insurance** | Alibi details, pre-emptive explanations, value inflation |

## Quick Start

### Run Locally

```bash
pip install flask
python app.py
```

Then open http://localhost:5000

### Deploy on Replit

1. Create a new Python Repl
2. Upload `app.py`, `requirements.txt`, and the `templates` folder
3. Click Run

## Usage

1. Paste text into the input box
2. Click "Analyze for Deception Indicators"
3. Review:
   - **Risk Score** (0-100)
   - **Highlighted Text** (hover for details)
   - **Indicators Found** (sorted by weight)
   - **Recommendations** (actionable next steps)

## Example Output

For a suspicious insurance claim:

```
SCORE: 24/100 (Medium Risk)
INDICATORS: 8 detected

- Convenient Documentation (+3.46)
- Value Inflation Signals (+2.98)
- Overly Specific Alibi (+1.70)
- Pre-emptive Explanation (+0.90)

RECOMMENDATION: Multiple insurance fraud indicators detected.
Recommend SIU referral and detailed investigation.
```

## Highlight Colors

- 🔴 **Red** - High weight indicators (1.5+)
- 🟠 **Orange** - Medium weight (1.3-1.5)
- 🟡 **Yellow** - Lower weight (<1.3)

## Based On

This tool is based on Forensic Statement and Linguistic Analysis (FSLA) principles, informed by research from [Deception Detection Lab](https://www.ddlltd.com).

## License

MIT License - Free for personal and commercial use.

## Contributing

Pull requests welcome. For major changes, please open an issue first.
