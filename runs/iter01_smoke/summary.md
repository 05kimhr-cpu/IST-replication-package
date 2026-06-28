# Iteration 1 smoke results
- language: py
- samples: 50

## Applicability (how often the perturbation produced a change)
- whitespace_double: 50/50
- trailing_period_flip: 50/50
- synonym_swap: 18/50
- action_antonym: 14/50

## Mean score vs identity (1.0) by (kind, perturbation, metric)

| kind | perturbation | metric | mean | median | min | n |
|------|--------------|--------|------|--------|-----|---|
| meaning_change | action_antonym | bleu | 0.8477 | 0.8633 | 0.6703 | 14 |
| meaning_change | action_antonym | rougeL | 0.8792 | 0.8819 | 0.8333 | 14 |
| paraphrase | synonym_swap | bleu | 0.8003 | 0.8482 | 0.2608 | 18 |
| paraphrase | synonym_swap | rougeL | 0.8665 | 0.8750 | 0.6667 | 18 |
| trivial | trailing_period_flip | bleu | 0.8909 | 0.8940 | 0.8091 | 50 |
| trivial | trailing_period_flip | rougeL | 1.0000 | 1.0000 | 1.0000 | 50 |
| trivial | whitespace_double | bleu | 1.0000 | 1.0000 | 1.0000 | 50 |
| trivial | whitespace_double | rougeL | 1.0000 | 1.0000 | 1.0000 | 50 |
