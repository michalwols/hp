"""SKETCH -- feedback with no absolute scale.

Image generation, writing quality and chat responses have no ground-truth
score. All you get is "A beat B", and latent quality has to be recovered from
comparisons -- LMArena-style Elo, or Bradley-Terry.
"""

import hp


class Sampler(hp.Params):
  steps: int = hp.IntRange(10, 100, default=30)
  guidance: float = hp.Range(1.0, 15.0, default=7.5)
  scheduler: str = hp.Choice(('ddim', 'dpm', 'euler'), default='dpm')


# --- A. comparisons as a distinct outcome channel ---------------------------

def compare(a: Sampler, b: Sampler, prompt: str) -> hp.Outcome:
  left, right = render(a, prompt), render(b, prompt)
  winner = human_or_judge(left, right)
  return hp.Outcome(
    comparisons=[hp.Prefer(winner=winner, loser=(b if winner is a else a))],
    feedback='B has better composition but washed-out colour',
  )


study = hp.optimize(
  Sampler,
  cases=prompts,
  compare=compare,                       # pairwise rather than absolute
  method=hp.optim.elo(rounds=200),       # fits latent scores from the pairs
)


# --- B. comparisons reduced to a score at the boundary ----------------------

def score_against_baseline(config: Sampler, prompt: str) -> float:
  """Win rate against a fixed reference -- an absolute number again."""
  wins = sum(human_or_judge(render(config, p), render(BASELINE, p)) is not BASELINE
             for p in prompt_batch)
  return wins / len(prompt_batch)


# NOTES ----------------------------------------------------------------------
#
# A. comparisons as their own channel
#   + honest about the data: nobody can score an image on an absolute scale,
#     but anyone can pick the better of two
#   + Bradley-Terry / Elo recovers a latent score, and the uncertainty on it,
#     which tells the optimizer where another comparison is worth buying
#   + the same pairs are exactly what DPO consumes, so a sweep and a
#     preference-tuning run can share one dataset
#   - most optimizers want a scalar, so anything numeric needs the fitted
#     latent score anyway
#   - O(n^2) pairs; needs active selection of which comparisons to run
#   - a trial's score changes as later comparisons arrive, so results are not
#     append-only -- that breaks the usual "trial is done" assumption
#
# B. reduce to a win rate against a baseline
#   + every existing optimizer works unchanged
#   + cheap, and easy to explain
#   - all information about *who* you beat is thrown away
#   - the baseline anchors everything: progress past it is invisible, and a
#     bad baseline quietly caps the whole study
#
# Suggested: support A as a real channel and provide B as a one-line adapter.
# The cost of A is not the Elo maths, it is that trial scores stop being final.
