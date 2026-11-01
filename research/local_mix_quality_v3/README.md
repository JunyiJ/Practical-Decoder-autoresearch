# Finite causal local-mixing comparison

Compare identity-initialized causal three-tap normalized FFN input mixing with
none on the adopted D256 MLA score0.5/SwiGLU672/SiLU/blockRMS/no product norm/
no warmup/H200 backbone. Candidate adds3072 parameters;88/100EMA tensors and
196608/204800B cache per batch item. Three equal peak LRs per arm, six1200-step
screens then three fresh pairs. Per-arm minimum full-validation H200 CE/lower
LR ties. No pilot-loss selection, cost-only adoption, retries or grid extension.
All521 predecessor sources unchanged. Use exact final freeze SHA and strict
four-thread MPS environment; complete retained-output allowance13555990528B
includes8GiB reserve and256MiB all-metadata/transient allowance. Exact pilot
retirement requires real quality integration; only four losing screen pairs
retire after complete independent audits and saved selection. All selected
and fresh models retained; holdout closed.
