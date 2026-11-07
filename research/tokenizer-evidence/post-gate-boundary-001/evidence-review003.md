# Readout-mixture algebra and shape-resource design completed

The exact three-arm native/MoC2/MoS2 design specifies tanh context maps, softmax
mixture weights, stable target-only probability-mixture loss, shared untied head,
independent standard private context initialization and zero-initialized R.
It resolves the symmetry issue without correlated maps or an initialization grid.
Both matched arms have131,584additional parameters, two matrixMuon maps plus
one fixed-output AdamW router. All old source/state/scorer consumers remain closed.

Only a tiny generated functional component is admitted next. Independent scalar
analytic gradients use mixture responsibilities, then explicit softmax/tanh/
shared-head chains. CPU/MPS guards and fixed tolerances precede first execution.
Full model/optimizer/RAW-H200/scorer/cache consumers remain unqualified. No model,
optimizer, corpus, checkpoint, seed or numerical quality work occurred here.

Complete future retention envelope26,845,642,752bytes,target28,993,126,400 is
covered by66,994,397,184free in a nonblocking external-lock pass. Zero deletion/
inventory, no old/future-retirement credit. This remains a prospective envelope;
counting-stream serialization and actual workers/pilots must qualify it before
any full scientific freeze. See TOKENIZER_READOUT_MIXTURE_PREPARATION.md and
readout-mixture-design001.json. No storage blocker or admitted quality panel.
