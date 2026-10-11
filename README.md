# PromoPilot

A working retail promotion planner with optional AI document interpretation and live tool coordination. It selects products and percentage, buy-two or bundle offers within inventory, margin, discount and advertising constraints. Results are simulations; campaigns are never launched.

## Installation — Python 3.11 or 3.12

Clone https://github.com/riyabisht30/PromoPilot or download and extract its source ZIP. Open a terminal in the folder containing app.py.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open http://localhost:8501. Select **Local engine (no API)** and click **Generate / replan**. The sample catalogue runs without a key. Runtime dependencies are listed in requirements.txt; no database or separate API server is required.

## Optional AI environment setup

Fresh document extraction and **Live Groq agent** require your own Groq key and model access. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and replace its placeholder key. PowerShell:

```powershell
Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Alternatively set GROQ_API_KEY, GROQ_MODEL and GROQ_VISION_MODEL environment variables. Model defaults are in the template and configurable in the app. Access/quota depend on your account. Never publish secrets.toml or a real key. Rotate any key previously exposed in an example/public file.

TXT/text-PDF extraction uses strict JSON Schema on supported text models, followed by semantic validation. Images use the existing vision path. Scanned PDFs require exporting a relevant page as PNG/JPG. Limits: 8 MB, at most 10 PDF pages, first 20,000 text characters; images are resized to at most 1600 pixels per side.

## How to use

1. Generate a local sample plan; inspect offers, constraints, feasible discounts and sensitivity.
2. Change advertising budget or apply a disruption and replan without an API. Restrict allowed mechanisms to compare percentage, buy-two and bundles. Bundle decisions use declared complementary pairs.
3. With AI access, upload data/policy.txt or data/competitor.png. Inspect extracted quotes, values and warnings; explicitly review, then apply.
4. Reuse reviewed evidence in local or live planning. Remove evidence to return to structured controls. Export the plan to inspect effective inputs and provenance.

Non-null reviewed policy limits override corresponding sidebar values; absent optional values retain settings. Budget-cut disruption applies afterward. Conflicts, missing-required values, invalid prices and unknown/duplicate SKUs block evidence application. Relevant evidence/source/model/schema changes require review again.

## Routes and safe failure

The local engine inspects, plans, validates and simulates without AI. Live Groq coordinates four fixed tools using the same constraints; the natural-language goal controls coordination, not arbitrary numerical settings. The visible quota option can fall back to local planning after HTTP429/quota failure. UI/export record actual local execution and ai_workflow_completed=false; the failed AI attempt remains recorded. Disable this option to stop and retain verified work. New document extraction has no fabricated-answer fallback. Other errors do not trigger quota fallback; retries/deadlines are bounded.

## Features and limitations

Seven audited features: product selection, mechanism selection, discount search, declared product relationships, component inventory constraints, competitor-price calculations and pre-launch simulation. Selection is finite candidate search plus a greedy heuristic, not proven global optimality. Demand is assumed; no measured causal uplift is claimed. Cannibalization, geographic customization and learned relationships are not implemented. F3 is supported in this scope; D2 remains provisional and D3 is unverified because AI interpretation is not consistently reliable. Local fallback does not prove AI extraction reliability.

## Architecture, API and verification

[ARCHITECTURE.md](ARCHITECTURE.md) explains flow and controls. [API.md](API.md) documents Python interfaces and JSON exports; there is no public REST API. app.py is the UI, engine.py handles calculations, agent.py integrates Groq, and workflow.py handles local execution/review/quota fallback.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

83 regressions cover calculations, evidence safeguards, provider handling and Streamlit UI/export behavior. Live provider calls are mocked in regressions; passing tests do not establish AI reliability. This judge-only source copy was tested with existing dependencies. A fresh installation and published-repository clean clone must still be checked.
