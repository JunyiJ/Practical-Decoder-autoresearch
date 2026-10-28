# Width512 attention pilots completed

Both frozen20-update pilots passed, source5824249 / plan7e592ec,
digest28887fbcbde92468ad9ed23b1537a2679d69591a7ec76cf8f9cfb99ffcc56d3c.
GQA then MLA; seed3391/LR0.00125; no selection from pilot scores.

Raw/H200 pairs: GQA1,212,185,804 bytes; MLA1,205,037,884 bytes, below frozen
allowances1,737,836,135/1,725,319,424. Largest process RSS2,449,309,696 bytes.
Measured parameter bytes251,510,784/249,696,768; cache per batch item/context128
1,048,576/327,680 bytes. Raw/H200 cached/full checks passed at batches1/8.

Hashes/config/tokenizer/20steps/3snapshots/finite model+optimizer+EMA and exact
EMA-export tensor equality passed. Both workers exited; no failure or retry.
Remaining12quality allowances plus8GiB reserve fit observed33,222,176,768 free
bytes. These short checks establish feasibility only, not long-training stability
or comparative quality. All checkpoint files remain in research_runs; portable
logs/results/requests and review are retained here. Test holdout remains closed.
