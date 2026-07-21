"""SKETCH -- hp as the config layer under a columnar tracking tool.

multilog logs into parquet / lance and queries with duckdb:

    import multilog as ml
    ml.log(params)
    ml.log.metric(loss=0.31, step=100)
    ml.query("select * from runs where config.optim.lr < 1e-4")

hp already produces everything such a tool needs; the question is what shape it
hands over.
"""

import hp


class Train(hp.Params):
  seed: int = 42
  epochs: int = 10
  lr: float = hp.LogRange(1e-6, 1e-2, default=3e-4, help='peak learning rate')
  api_key: str = hp.Field(default='', secret=True)
  model = object()                       # not columnar-storable


config = hp.load(Train, 'base.yaml', hp.cli)


# --- what hp hands over -----------------------------------------------------

hp.serializable(config)     # {'seed': 42, 'lr': 0.0003, 'model': 'torch.nn.Linear'}
                            # secrets dropped, unstorable values named not pickled
hp.flatten(config)          # dotted paths, one column each
hp.sources(config)          # {'lr': 'cli', 'seed': 'base.yaml', ...}
hp.stable_hash(config)      # content-addressed run id
hp.field_paths(config)      # types, ranges, units, help -> a real schema


# --- A. wide config, long metrics -------------------------------------------
# The classic split, and what wandb does.

ml.log(config)                                  # one row in `runs`
ml.log.metric(loss=0.31, accuracy=0.88, step=100)   # many rows in `metrics`

ml.query("""
  select r.config.lr, min(m.value) as best_loss
  from runs r join metrics m using (run_id)
  where m.key = 'loss'
  group by 1 order by 2
""")


# --- B. config as a nested struct vs flattened columns ----------------------

ml.log(hp.serializable(config))    # struct column: config.optim.lr
ml.log(hp.flatten(config))         # flat columns: "optim.lr"


# --- C. provenance as its own table -----------------------------------------
# Nothing else logs this, and it answers the question people actually ask.

ml.log.sources(hp.sources(config))

ml.query("""
  select path, source, count(*)
  from run_sources where source = 'cli'
  group by 1, 2                     -- what do people actually override?
""")


# NOTES ----------------------------------------------------------------------
#
# What hp should hand over
#   + hp.serializable is the safe default: secrets dropped, unstorable values
#     replaced by their type name rather than crashing the parquet writer or
#     quietly pickling a whole model
#   + hp.flatten now omits secrets too -- it did not, and a logger reaching for
#     the obvious columnar shape would have written the API key to disk
#   + params expose _asdict(), so a tool that already reads NamedTuples and
#     dataclasses picks them up without importing hp at all
#
# A. wide config + long metrics
#   + metrics vary per run and are sparse; long format absorbs that without
#     schema churn, and duckdb pivots on read
#   + config is naturally one row per run
#   - two tables means every question is a join; fine in duckdb, annoying in
#     a notebook
#
# B. struct vs flattened config
#   struct  + survives adding and removing fields without column explosion
#           + duckdb addresses nested fields directly: config.optim.lr
#           - awkward to select * and eyeball
#   flat    + trivially readable, one column per knob
#           - a hundred configs with different fields means a hundred columns,
#             most of them null, and renaming a field forks the schema
#   Struct is the better default; flatten is the better view.
#
# The schema problem is the real one
#   - runs from different code versions have different fields. Parquet handles
#     it per file and duckdb reads across them with union_by_name, but any
#     query written against yesterday's columns silently returns nulls today.
#   - hp.field_paths gives types, ranges and help text, so multilog could
#     record the *schema* alongside the values and detect drift rather than
#     discovering it in a wrong answer
#
# Identity
#   + hp.stable_hash is a content-addressed id: the same config always hashes
#     the same, which makes dedup and "have I already run this?" a lookup
#   - it hashes config only. Two runs with identical config and different code
#     collide, so the run key wants to be (config_hash, code_version)
#
# Lance vs parquet
#   + parquet for runs, metrics and sources -- append-mostly, columnar scans
#   + lance for traces, embeddings and artifacts -- random access and blobs,
#     which is exactly what an agentic optimizer replays
