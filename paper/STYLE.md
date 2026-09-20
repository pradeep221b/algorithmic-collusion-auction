# Paper conventions (binding for every writer and checker)

Working title: **Collusion or frozen learning? Independent Q-learners in a uniform-price electricity auction**
Author: Pradeep Sriramula (independent). Target: a 15–20 page working paper (SSRN/arXiv style), economics
of algorithmic pricing / power-system economics audience. Voice: plain, precise, no hype.

## The one rule
Every number in the paper comes from `paper/FACTS.md`, which is derived from `results/*.json`.
No number may be invented, rounded differently, or "remembered". If a fact you need is not in
FACTS.md, write `% TODO-FACT: <what you need>` in the .tex and say so in your report. Do not guess.

## Claims discipline
- "supra-competitive prices" = observed fact. "tacit collusion" = a *claim* that needs the indicators.
  Never write that the agents collude. The paper's thesis is that the standard indicators cannot
  support that claim here, and that one Q-table test can separate learned continuation from a
  stale estimate.
- State the two failed pre-registered predictions plainly (Table `tab:predictions`).
- Every experiment has 100 seeds unless FACTS.md says otherwise. CI = mean ± 1.96 × SE across seeds.

## LaTeX
- `\documentclass[11pt,a4paper]{article}`; packages already loaded in `main.tex`: amsmath, amssymb,
  booktabs, graphicx, natbib (authoryear, round), hyperref, tikz, caption, subcaption, xcolor, siunitx.
- Section files live in `paper/sections/` and are `\input` by `main.tex`. Write ONLY your file. No
  `\documentclass`, no preamble, no `\begin{document}` in section files.
- Macros available: `\Dl` = collusion index Δ (typeset `$\Dl$`); `\eur` = € sign (`\eur{100}`);
  `\MW` (`\SI{60}{\MW}` works via siunitx: use `\qty{60}{\mega\watt}` or write "60 MW" in text).
  Keep it simple: "60 MW" in text; use `\eur{...}` for every euro amount, in text and in math.
- Tables: `booktabs` (`\toprule/\midrule/\bottomrule`), no vertical rules, numbers right-aligned,
  caption ABOVE the table, one sentence of caption then a "Notes:" sentence if needed.
- Figures: `\includegraphics[width=\textwidth]{figures/<file>}`, caption BELOW.
- Citations: natbib. `\citet{key}` for "Calvano et al. (2020) show", `\citep{key}` for "(Calvano et al., 2020)".
  Only keys from `paper/refs.bib`. Never invent a key.
- Cross-refs: `\Cref` is NOT loaded; use `Section~\ref{sec:model}`, `Table~\ref{tab:m2}`, `Figure~\ref{fig:phase1}`.

## Label registry (use exactly these; define each label once, in the file that owns it)
Sections: sec:intro (01), sec:related (02), sec:model (03), sec:design (04), sec:results (05),
sec:phase1 (06), sec:discussion (07), sec:conclusion (08), app:repro, app:params, app:extra (appendix.tex).
Subsections in 03: sec:market, sec:learners, sec:indicators. In 06: sec:beta, sec:reinject, sec:residual,
sec:payasbid, sec:occupation.
Tables: tab:market (03), tab:nash (03), tab:params (04), tab:m2 (05), tab:m3 (05), tab:m4 (05), tab:m5 (05),
tab:predictions (06), tab:sweep (06), tab:reinject (06), tab:residual (06), tab:payasbid (06),
tab:occupation (06), tab:indicators-summary (07).
Figures: fig:pipeline (03, TikZ), fig:trajectory (05, figures/price_trajectory.png), fig:trajectory5m
(05, figures/price_trajectory_long.png), fig:ablation (05, figures/ablation_bars.png), fig:punishment
(05, figures/punishment_long.png), fig:phase1 (06, figures/phase1.png), fig:mechanism (07, TikZ).
Equations: eq:qupdate (03).

## File ownership
01_intro.tex, 02_related.tex, 03_model.tex, 04_design.tex, 05_results.tex, 06_phase1.tex,
07_discussion.tex, 08_conclusion.tex, appendix.tex, abstract.tex (abstract only, ~180 words).

## Notation
Firms i = 1,2,3 (in text; code uses 0,1,2 — the paper uses 1,2,3 and says so once in the appendix).
Bid grid rungs r = 1..15, price p_r = 10 + (r−1)·90/14 €/MWh. Clearing price p. Collusion index
Δ = (p̄ − 10)/(100 − 10). Learning rate α, discount γ, exploration ε_t = e^{−βt}, memory m ∈ {0,1}.
Q-table Q_i(s, a). Horizon T ∈ {1M, 5M} rounds. "Frozen play" = greedy play with learning and
exploration off. "Converged" = no greedy action changed in the last 100,000 rounds.

## Length targets (words)
abstract 180 · intro 900 · related 900 · model 1100 · design 600 · results 1500 · phase1 1800 ·
discussion 1100 · conclusion 300 · appendix 500.
