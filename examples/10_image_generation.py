"""SKETCH -- optimizing an image generation stack.

The hard case: no absolute quality score exists, every cheap verifier is
gameable on its own, and the only ground truth is expensive human judgement.
The shape of the answer is a fidelity ladder -- cheap verifiers screen, humans
decide -- plus enough diversity in the verifiers that gaming one does not win.
"""

import hp


# --- the problem ------------------------------------------------------------

class Sampler(hp.Params):
  steps: int = hp.IntRange(4, 60, default=30)
  guidance: float = hp.Range(1.0, 12.0, default=6.0)
  scheduler: str = hp.Choice(('ddim', 'dpm++', 'euler_a'), default='dpm++')

  use_refiner: bool = False
  refiner_strength: float = hp.Range(0.0, 0.5, default=0.2,
                                     when=lambda root: root.use_refiner)

  # text, rewritten by a reflective optimizer rather than sampled
  negative_prompt: str = hp.Evolve(
    'blurry, watermark, extra fingers, text artifacts',
    description='Steers away from common failure modes',
  )
  style_suffix: str = hp.Evolve('', description='Appended to every prompt')

  # derived, so the sweep never searches compute directly
  total_steps = hp.Derived(
    lambda p: p.steps + (int(p.steps * p.refiner_strength) if p.use_refiner else 0),
  )

  # measured
  preference = hp.Objective(maximize=True)      # latent quality, from comparisons
  latency_ms = hp.Objective(minimize=True)
  diversity = hp.Metric()                       # watched, never optimized


# --- cheap verifiers: fast, noisy, individually gameable ---------------------

def alignment(image, prompt) -> float:
  """Does it depict what was asked? Alone: rewards literal, prompt-stuffed images."""
  return siglip_score(image, prompt)


def aesthetic(image) -> float:
  """Alone: rewards oversaturated, over-smoothed slop. Never use as the only signal."""
  return aesthetic_predictor(image)


def artifacts(image) -> hp.Outcome:
  found = detect_artifacts(image)          # extra limbs, garbled text, seams
  if not found:
    return hp.Outcome(score=1.0)
  return hp.Outcome(score=0.0, feedback=f'artifacts: {", ".join(found)}')


def safety(image) -> hp.Outcome:
  """A gate, not a trade-off -- reported as a constraint, never weighted."""
  return hp.Outcome(score=0.0, constraints={'unsafe': 1.0 if nsfw(image) else -1.0})


def batch_diversity(images) -> float:
  """weight=0. Rises before quality collapses, so it is the early warning."""
  return mean_pairwise_distance(embed(images))


# --- human feedback: expensive, pairwise, the actual ground truth ------------

def human_compare(a: Sampler, b: Sampler, prompt: str) -> hp.Outcome:
  left, right = render(a, prompt), render(b, prompt)
  winner = cleanser_ab_prompt(left, right)      # a person, in cleanser
  return hp.Outcome(
    comparisons=[hp.Prefer(winner=winner, loser=b if winner is a else a)],
    feedback='B has better composition, washed-out colour',
  )


# --- the ladder -------------------------------------------------------------
# Fidelity here is feedback *quality*, not eval size: hundreds of configs are
# screened by verifiers, only survivors are shown to a person.

study = hp.optimize(
  Sampler,
  cases=prompts,                              # stratified from production traffic
  verifiers=[alignment, aesthetic, artifacts, safety, batch_diversity],
  weights=[1.0, 0.4, 0.6, 0.0, 0.0],
  constraints={'unsafe': 0.0, 'latency_ms': 4000},
  method=hp.optim.tpe(trials=400, pruner='asha'),
)

# only the Pareto frontier is worth a human's time
finalists = study.pareto[:12]

elo = hp.optimize(
  finalists,
  cases=prompts,
  compare=human_compare,
  method=hp.optim.elo(budget=300, select='uncertainty'),   # buy the informative pairs
)


# --- what the comparisons are worth twice -----------------------------------
# The same pairs that ranked the configs are Diffusion-DPO training data.

pairs = hp.params.history('human_compare')
yann.train_dpo(model, pairs, config=elo.best)

# and distillation is reference-as-constraint: match the 50-step render, race on time
distilled = hp.optimize(
  Sampler,
  cases=prompts,
  verifiers=[hp.reference(render_50_step, equivalent=lpips_below(0.15)), latency],
  weights=[0.0, 1.0],
  constraints={'mismatch_rate': 0.05},
  method=hp.optim.tpe(trials=200),
)


# NOTES ----------------------------------------------------------------------
#
# Why a ladder rather than one objective
#   + human judgement is the only ground truth and costs orders of magnitude
#     more than a verifier; screening 400 configs down to 12 before spending it
#     is the difference between feasible and not
#   + fidelity as *feedback quality* maps onto ASHA unchanged -- cheap rung,
#     expensive rung, promote survivors
#   - agreement with cheap verifiers correlates imperfectly with human
#     preference, so the frontier handed up is only as good as the screen
#
# Verifier diversity is the anti-hacking mechanism
#   + aesthetic predictors alone produce the familiar over-smoothed look;
#     alignment alone produces literal prompt-stuffed images. Optimizing several
#     signals that fail differently is what stops any one being gamed
#   + `diversity` at weight 0 is the tripwire: it drops before quality visibly
#     collapses, and it costs nothing to watch
#   + a held-out human eval the optimizer never sees is the only real check
#   - more verifiers means more weights, and the weights are themselves
#     unknown -- which is the argument for optimizing the Pareto front and
#     choosing an operating point afterwards rather than guessing a scalar
#
# Safety is a constraint, not a term
#   + weighting it means a large enough quality gain can buy an unsafe image.
#     As a constraint it is simply infeasible
#   - constraints need the sampler to support them (TPE, GP, NSGA-II do)
#
# Comparisons are dual-use
#   + every pair a human labels both ranks the configs and becomes a DPO
#     training pair. The eval loop and the training set are the same artifact,
#     which is the "one substrate" claim made concrete
#   - a trial's Elo moves as later comparisons land, so trial scores are not
#     final and the store must tolerate updates
#
# Text components are evolved, not sampled
#   + a negative prompt is not a categorical variable; there is no list to pick
#     from. Marking it Evolve hands it to a reflective optimizer that reads the
#     artifact feedback and rewrites it
#   - that only works if verifiers return *why* they failed, not just a score
