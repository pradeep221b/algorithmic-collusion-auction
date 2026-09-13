# Reading plan — from zero to defending this project

Ordered so each layer explains the next. Print in this order. "Free" means a legal copy the author
or publisher put online. "Paywalled" means buy the book or use Google Scholar → "All versions" to
find an author-hosted preprint. Time estimates assume 4–5 focused hours a day.

After each item there is a list of questions. Do not move on until you can answer them out loud
without notes. Those are the questions a professor will ask.

---

## 0. How to read a paper (day 1)

Read in three passes. Pass 1: abstract, introduction, figures, conclusion, 20 minutes; write one
sentence: what did they find. Pass 2: the model section and the main results table, 1–2 hours;
write down every symbol and what it means. Pass 3: only for the six core papers in §4 below, read
every line and every footnote, redo one number by hand.

- Keshav, "How to Read a Paper" (free, 2 pages): search the title, it is on the author's Waterloo page.

---

## 1. Electricity markets — what the agents are bidding into (week 1)

**1.1 Cramton, "Electricity market design", Oxford Review of Economic Policy 2017.** Free:
http://www.cramton.umd.edu/papers2015-2019/cramton-electricity-market-design.pdf
Read all of it. 25 pages.
- What is a day-ahead market, a real-time market, a uniform-price auction?
- Why is every dispatched firm paid the marginal bid and not its own bid?
- What is "pay-as-bid" and why do most markets not use it?

**1.2 PJM "Learning Center" and ERCOT "Nodal 101" primers.** Free web pages, search the names.
Skim, one hour. Know what an ISO/RTO is and that real markets have thousands of bidders, network
constraints, and multi-segment bid curves. Our model has none of these; know why (SPEC.md).

**1.3 Stoft, *Power System Economics* (Wiley/IEEE, 2002).** Paywalled book, about $100, worth
it if you stay in this field. Part 4 (market power) is the part you need. Kirschen & Strbac,
*Fundamentals of Power System Economics*, is the shorter alternative.
- What makes a supplier "pivotal"? Why does our spec choose demand = 100 MW with three 60 MW firms?
- What is the competitive benchmark price in our model and why is it exactly marginal cost?
- Why is the clearing price in our auction always the second-lowest bid?

---

## 2. Game theory — what "collusion" means precisely (weeks 2–3)

**2.1 Osborne & Rubinstein, *A Course in Game Theory* (MIT Press 1994).** Free full text from the
author: https://www.economics.utoronto.ca/osborne/cgt/INDEXR.HTM
Read chapters 1, 2 (strategic games, Nash equilibrium), 8 (repeated games). Skip the rest.
- What is a Nash equilibrium? Why is bidding at marginal cost the one-shot Nash of our auction?
- What is the Bertrand paradox?
- What is the folk theorem? Why does it need a discount factor close to 1 and memory of the past?
- What is a trigger strategy? A grim trigger? Punish-then-forgive?

**2.2 Ivaldi, Jullien, Rey, Seabright, Tirole, "The Economics of Tacit Collusion" (2003).** Free:
report written for the European Commission, search the title, it is on the IDEI Toulouse site
and on the EC competition site. 75 pages, read sections 1–3.
- What is the difference between explicit and tacit collusion?
- What market features make tacit collusion easier? Which of them does our model have?

**2.3 Tirole, *The Theory of Industrial Organization* (MIT Press 1988), chapter 6.** Paywalled
book. The standard reference. Optional if 2.1 and 2.2 are solid.

**2.4 Maskin & Tirole, "A Theory of Dynamic Oligopoly II: Price Competition, Kinked Demand Curves,
and Edgeworth Cycles", Econometrica 1988.** Paywalled (JSTOR). Google Scholar → All versions
usually finds a copy. Read sections 1–3 only. This is the mechanism behind our memory-0 result.
- What is an Edgeworth cycle? Draw one.
- Why is undercutting a chain of steps but raising a single jump?

**2.5 Young, "The Evolution of Conventions", Econometrica 1993.** Free copy:
http://dklevine.com/archive/refs4485.pdf
Read introduction and section 2 (the idea, not the proofs). Kandori, Mailath & Rob 1993 is the
companion paper; same archive, search "Learning, Mutation, and Long Run Equilibria in Games".
- What is stochastic stability? What does "as the noise goes to zero" select?
- In our PHASE1.md §5, which learner's frozen price is a small-noise limit, and how do we know?

---

## 3. Reinforcement learning — what the agents are (week 3–4)

**3.1 Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (MIT Press 2018).** Free
from the authors: http://incompleteideas.net/book/the-book-2nd.html
Read chapters 1, 2 (bandits, ε-greedy), 3 (MDPs), 6.1–6.5 (TD learning, Q-learning). Do the
exercises in 6.5. Skip everything on function approximation for now.
- Write the Q-learning update from memory. What are α, γ, ε? What does each control?
- What is a state, an action, a reward in *our* model? Why is the state last round's bids?
- Why does exploration have to decay? What breaks if it decays too fast? Too slow?
- What is a Bellman backup? What does a Bellman residual measure? (This is our §3 indicator.)
- What is "off-policy"? Why does Q-learning learn about actions it did not take, and why does
  that fail for actions it *never* takes?

**3.2 David Silver, UCL RL lecture series (2015).** Free on YouTube. Lectures 1–5. Watch after
reading 3.1, not instead of it.

**3.3 Multi-agent reality check.** Sutton & Barto assume one agent and a fixed environment. With
three learners the environment is *not* fixed, and Q-learning has no convergence guarantee.
Read: Busoniu, Babuska & De Schutter, "A Comprehensive Survey of Multiagent Reinforcement
Learning", IEEE SMC-C 2008, free preprint on the TU Delft site. Sections 1–3.
- Why does independent Q-learning have no convergence guarantee with several learners?
- What does "converged" mean in our code (agents.py CONV_ROUNDS) and why is it a weak notion?

---

## 4. Algorithmic collusion — the literature this project speaks to (weeks 5–6)

Read in this order. Pass 3 (every line) for 4.1, 4.5, 4.6.

**4.1 Calvano, Calzolari, Denicolò, Pastorello, "Artificial Intelligence, Algorithmic Pricing,
and Collusion", AER 2020.** Free copy from the Bologna repository:
https://cris.unibo.it/bitstream/11585/832496/7/Algorithmic%20collusion.pdf
The paper that started the field. Our Δ, our Q-initialisation, our ε schedule, our punishment
test are all theirs.
- What is Δ? What are the competitive and collusive benchmarks in their model versus ours?
- How do they test for punishment? How do they show the strategies are "collusive" rather than
  merely high-priced? Which of those tests did our RESULTS.md replicate?
- What is their finding about memory and discount factor? How does ours differ?

**4.2 Klein, "Autonomous Algorithmic Collusion: Q-Learning under Sequential Pricing", RAND 2021.**
Free working paper: https://ssrn.com/abstract=3195812
- What changes when firms move in turns instead of together? Why does that connect to Edgeworth cycles?

**4.3 Asker, Fershtman, Pakes, "Artificial Intelligence and Pricing: The Impact of Algorithm
Design", NBER WP 28535 (2021).** Free: https://www.nber.org/papers/w28535
- What is synchronous versus asynchronous learning? Which is ours?
- Why does the learning *protocol*, not the firms' incentives, decide the price?

**4.4 Banchio & Skrzypacz, "Artificial Intelligence and Auction Design" (2022).** Free:
https://arxiv.org/abs/2202.05947
The one paper on Q-learners in *auctions*.
- Why do first-price auctions give low bids and second-price auctions do not?
- Our uniform-price auction: which of the two is it closer to, and why?

**4.5 Abada & Lambin, "Artificial Intelligence: Can Seemingly Collusive Outcomes Be Avoided?",
Management Science 2023.** Free working paper:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3559308
The paper closest to our Phase 1. Electricity storage, Q-learners, and the finding that
"collusion" came from imperfect exploration.
- What is their mechanism? How is it the same as our §5, and how is it different?
- What do they propose to fix it? Would it fix our memory-0 freeze? (Our β sweep says no; why?)

**4.6 Abada, Lambin, Tchakarov, "Collusion by Mistake: Does Algorithmic Sophistication Drive
Supra-Competitive Profits?", EJOR 2024.** Free working paper:
https://ssrn.com/abstract=4099361
- What does "collusion by mistake" mean? Is our memory-0 result an instance of it?
- What would they say our Bellman-residual indicator measures?

**4.7 Bichler et al., "Algorithmic Pricing and Algorithmic Collusion" (survey, 2025).** Free:
https://arxiv.org/pdf/2504.16592
Read *last*, as a map of everything else. Note every paper it cites that you have not read; that
is your Phase 2 reading list.

**4.8 Harrington, "Developing Competition Law for Collusion by Autonomous Artificial Agents",
Journal of Competition Law & Economics 2018.** Search SSRN for the title. The legal side: what
would a court need to see. Skim.

---

## 5. Policy — why anyone outside academia cares (week 7, two days)

**5.1 OECD, "Algorithms and Collusion: Competition Policy in the Digital Age" (2017).** Free:
https://www.oecd.org/content/dam/oecd/en/publications/reports/2017/05/algorithms-and-collusion-competition-policy-in-the-digital-age_02371a73/258dcb14-en.pdf
Chapters 1–3.

**5.2 UK CMA, "Algorithms: How they can reduce competition and harm consumers" (2021).** Free on
gov.uk, search the title. Section on pricing algorithms.

**5.3 ACER, REMIT (EU regulation on energy market integrity).** Know that it exists, that energy
regulators already screen bids for manipulation, and that "algorithmic bidding" is on their agenda.
Search "ACER REMIT algorithmic trading guidance".
- If a regulator used our M3–M5 indicators on real bid data, what would go wrong? (PHASE1.md says.)
- What could a regulator observe in a real market that we can observe in a simulation? What can't they?

---

## 6. Code and tools — what already exists (week 7, three days)

- **ASSUME** (INATECH Freiburg): https://github.com/assume-framework/assume — agent-based electricity
  market simulator with RL bidders. Read the README and the docs on RL agents. Know what it does
  that we do not, and what we do that it does not.
- **Gymnasium** and **PettingZoo** docs (Farama Foundation): the standard single- and multi-agent
  environment interfaces. Read the "basic usage" pages. Phase 2 packages our simulator to this API.
- **Grid2Op** (RTE): grid-operation RL benchmark, the L2RPN competition. Skim the README only.
- Calvano et al. published their Fortran code with the AER paper (data appendix on the AEA site).
  Open it once. See how much of it our 150-line agents.py replaces.

---

## 7. Your own project — read it like a stranger (week 8)

In this order: SPEC.md, market.py, agents.py, tests/, RESULTS.md, phase1.py, PHASE1.md.
For every function: what goes in, what comes out, why it exists. For every table: pick one number
and regenerate it from the JSON in results/.

Then write the one-page summary I asked for: what the memory-0 learner does, what the memory-1
learner does, how we know. If you can write that page and answer every question above, you can
defend this project to anyone.

---

## Order and time

| week | read | outcome |
|---|---|---|
| 1 | §0, §1 | can explain the auction to a friend in 5 minutes |
| 2–3 | §2 | can define Nash, folk theorem, tacit collusion, Edgeworth cycle, stochastic stability |
| 3–4 | §3 | can write the Q-learning update and explain every line of agents.train |
| 5–6 | §4 | can place our result against each of the six core papers in one sentence each |
| 7 | §5, §6 | knows why a regulator cares and what tools exist |
| 8 | §7 | one-page summary written, sent to me |

Eight weeks full-time. It will feel slow in weeks 2–3. That is the part that cannot be skipped.
