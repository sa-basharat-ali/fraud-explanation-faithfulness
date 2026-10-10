# Fraud explanation faithfulness

```
python3 explain_eval.py          # stdlib only: scores the 80 cached explanations
python3 explain_eval.py --ask    # optional: asks Claude again (Claude Code CLI)
```

Fraud models now ship natural-language explanations next to every score, written for the analyst who
makes the call. Does the explanation tell the truth about the model?

A synthetic identity-fraud book of 30,000 applications and a logistic model (AUC about 0.95). Because
the model is linear, every score decomposes exactly into per-feature contributions, so its real top-3
reasons are known, with no attribution approximation to argue about. Some features look suspicious to a
human and barely matter to the model (a VoIP phone, a far-away IP, a name that isn't in the email
handle); others look dull and drive the score (phone tenure, email age, applications sharing a device).
Fixed seed; no real person or customer.

Claude (Haiku, through the Claude Code CLI) wrote the explanation for 40 flagged applications under two
prompts, citing its reasons from a fixed list of fields:

| prompt | cited reasons | in the model's top 3 | real but minor (#4 or lower) | the model counted against fraud, or ignored |
|---|---|---|---|---|
| open: sees the data and the score | 240 | 111 | 107 | **22** |
| grounded: also given the model's top 3, told to use only those | 120 | 120 | 0 | **0** |

**16 of 40 open explanations (40%) cite at least one reason the model held against fraud or ignored.**
In one, the model's top reasons were SSN prefix reuse, email age and a shared device; the explanation
also blamed the phone, whose long tenure was actually pulling the score down. The analyst reads a
reason the model never used, and if they approve or decline on it, the decision rests on a story the
model didn't tell. Grounding fixes it: 0 of 40.

## The part I got wrong

My first scoring counted every cited reason outside the model's top 3 as unfaithful, which said 40 of 40
open explanations were wrong and 129 of 240 reasons were invented. That was too harsh: 107 of those
were real contributions, just smaller ones. Splitting "real but minor" from "the model held it against
fraud" gave the honest number, 16 of 40. A keyword scan of the narratives also flagged 8 grounded
explanations, all false alarms: "email address" matched my pattern for the address-velocity feature.

## Files

| | |
|---|---|
| `model.py` | applications, features, the logistic model, exact per-feature contributions |
| `explain_eval.py` | 40 flagged cases, two prompts, scoring each cited reason against the model's contributions |
| `answers/` | the 80 explanations Claude wrote, cached |

Syed Ahmed Basharat Ali · sabasharat.ali@gmail.com · basharat.net
