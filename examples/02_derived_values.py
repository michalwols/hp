"""Derived, Computed and property, and when to use which. RUNNABLE."""

from datetime import datetime, timezone

import hp


class Batch(hp.Params):
  micro: int = hp.Choice((1, 2, 4, 8), default=2)
  accum: int = hp.IntRange(1, 32, default=8)
  world_size: int = 1

  # (a) Derived: recomputed on every read, serialized, never searched
  global_batch = hp.Derived(lambda p: p.micro * p.accum * p.world_size)

  # (b) decorator form, for anything longer than a lambda
  @hp.derived
  def tokens_per_step(self):
    """Sequence tokens consumed per optimizer step."""
    return self.global_batch * 2048

  # (c) Computed: evaluated once per object, then fixed
  started_at = hp.Computed(lambda p: datetime.now(timezone.utc).isoformat())

  # (d) plain property: not a field, so it never serializes
  @property
  def label(self) -> str:
    return f'bs{self.global_batch}'


def main() -> None:
  batch = Batch()
  print('global_batch     ', batch.global_batch)
  batch.accum = 16
  print('after accum=16   ', batch.global_batch, '(recomputed)')
  print('tokens_per_step  ', batch.tokens_per_step)
  print('label (property) ', batch.label)

  print('computed stable  ', batch.started_at == batch.started_at)
  print('per instance     ', Batch().started_at != batch.started_at)

  print('serialized       ', sorted(hp.to_dict(batch)))
  print('search dims      ', sorted(hp.space(batch)))
  print('dependencies     ', hp.fields(batch)['global_batch'].dependencies(batch))

  try:
    batch.global_batch = 99
  except AttributeError as error:
    print('not settable     ', error)


if __name__ == '__main__':
  main()

# NOTES ----------------------------------------------------------------------
#
# Derived   + recomputes, so it can never disagree with its inputs
#           + serializes, so a saved config records what was actually used
#           + excluded from the search space, which is the point: searching
#             micro, accum AND global_batch wastes a sweep on a redundant axis
#           - a read costs a function call; fine for config, not for hot loops
#
# Computed  + stable for the life of the object, which is what a timestamp,
#             run id or hostname needs
#           + lazy, so nothing is paid for if never read
#           - "once" means first read, not construction; if you need the
#             construction instant exactly, pass it in instead
#
# property  + zero new concepts, familiar to every Python reader
#           - invisible to to_dict, save, diff and stable_hash, so two runs
#             that differ only in a property look identical
#
# Rule of thumb: if a future reader of the saved config would want the value,
# it is a Derived. If it only matters in memory, a property is enough.
