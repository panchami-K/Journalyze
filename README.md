# Journalyze (Journal + Analyze): Automated Self-Insight from Journaling

## Overview

**Journalyze** is an advanced, modular Python toolkit to uncover deep psychological patterns in your journal entries. By combining state-of-the-art NLP, emotion detection, cognitive bias analysis, and personality trait inference, Journalyze helps users transform daily writing into actionable self-awareness and growth.

Whether you're an individual seeking personal insight or a professional supporting others, Journalyze automates the process of cleaning, analyzing, and summarizing journal data to surface emotional cycles, cognitive distortions, trait-based tendencies, and recurring behavioral themes.

---

## Features

- **Text Cleaning & Anonymization:**  
  Safeguards sensitive information (names, emails, orgs) using regex/NLP.
- **Emotion & Bias Detection:**  
  Spots a rich palette of emotional states and cognitive distortions (e.g., overgeneralization, catastrophizing, mind reading).
- **Pattern & Loop Analysis:**  
  Quantifies recurring themes, emotional loops, and self-defeating patterns over time.
- **Personality Trait & Peer Group Profiling:**  
  Infers Big Five personality facets; groups users by peer similarity.
- **Insightful Period Summaries & Feedback:**  
  Generates monthly feedback on major emotional/bias trends and personalized growth advice.
- **Advanced Feature Engineering:**  
  Annotates entries with context triggers, textual stats, trait scores, and cluster IDs.
- **Visual Analytics (optional):**  
  Plots emotion/bias frequency, heatmaps, and behavioral clusters.
- **Business Metric Validation:**  
  Code samples for measuring review speed, precision/recall, and user engagement.

---

## Project Structure

journalyze/
│
├── main.py # Pipeline entry (run this file)
├── requirements.txt # All dependencies
├── config.yaml # Config file for data paths, settings
├── src/
│ ├── data_preprocess/ # Data loading, cleaning, anonymization
│ ├── eda/ # Exploratory analysis/reporting
│ ├── features/ # Feature engineering modules
│ ├── insights/ # Insight/feedback generators
│ └── visualization/ # (Optional) plotting code
├── data/ # Place your raw and processed CSVs here
└── README.md


---

## Getting Started

### 1. Clone and Install Requirements

git clone  https://github.com/panchami-K/Journalyze

cd journalyze
pip install -r requirements.txt


### 2. Prepare Data

- Place your journal CSVs in the `data/` folder (or as set in `config.yaml`).

### 3. (Optional) Edit Config

- Open and adjust `config.yaml` for custom paths or feature parameters.

### 4. Run the Main Pipeline

python main.py

*This cleans and analyzes your entries, runs pattern detection, clusters, and outputs results to `/data/processed/`.*

---

## Usage

- **Run end-to-end analysis:**  
  Executes all stages to produce a cleaned, feature-rich CSV and feedback summaries.
- **Visualize patterns:**  
  Extend with modules in `/src/visualization/` for emotion and bias trends.
- **Review insights:**  
  See terminal output and inspect saved CSV feedback for personal reflection.

---
## License

This project is licensed under the MIT License.



