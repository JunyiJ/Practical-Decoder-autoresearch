# Shape and stride diagnostic

This separate diagnostic executed no model candidate. It compared every output
row count1–259 at reduction lengths1,17,2048 with two fixed upstream patterns.
For each case, the259-row reference was compared with a compact allocation and
a prefix view retaining the original leading stride. All logical entries were equal.

All K=1 cases were exact. At K=17, only row count1 differed. At K=2048, row counts
1–14 differed and15–259 were exact. Retaining stride259 produced the same mismatch
sets. Maximum absolute differences were4.7684e-7 for K17 and7.6294e-6 for K2048.
These1554cases/3108comparisons support a shape-dependent numerical explanation,
not a memory-stride-only explanation or a universal exactness theorem.

The pinned [PyTorch source](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/LinearAlgebra.mm)
explicitly routes one-row outputs through GEMV. Other MPSGraph kernel choices
were not traced. Runtime: torch2.14.1, commit5c4886908584029761b579af026dcfb627c84070,
macOS26.3. The source notes and local header receipts are retained.

The guard finished in7.525seconds, peak RSS558,317,568bytes. No random roles,
training updates, quality scores or weights were created. The old failed
active-row prototype remains closed and unadopted.
