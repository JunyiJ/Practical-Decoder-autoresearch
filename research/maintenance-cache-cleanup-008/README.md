# Approved unused Docker image removal

The human explicitly identified browserless/chrome and the old openclaw-docker image plus its two inventoried untagged descendants as unused. The validated pass removed only those four exact SHA256 IDs through Docker, without force or parent pruning. Every removal had an immediate identity/container check, a durable intent/result journal, and the nonblocking external maintenance lock. Both running containers, their start times and mounts, all volumes, and every non-allowlisted image were preserved.

Docker released 8,515,156,486 unique layer bytes. After delayed automatic sparse-disk reclamation, the VM backing allocation fell by 4,992,946,176 bytes and host free space rose by 4,989,054,976 bytes. APFS and concurrent writes explain why these measurements need not agree. No VM backing file was directly modified, and no app, service or container was stopped or restarted.

Fresh free space was 13,666,439,168 bytes, above the 13,196,505,088-byte prepared-integration cleanup target by 469,934,080 bytes. The target includes both arms, retained RAW/H200 envelopes, metadata, the 8 GiB reserve, measured extra filesystem headroom and 2 GiB cleanup headroom. The future scientific panel is still unmeasured; this is not a claim that it fits.

Resume state was reconciled under the shared worker lock with a concurrent-change check. The next research continuation must remeasure and validate bindings before the separately prepared resume002; closed resume001 must never be retried. No experiment was launched here. The existing maintenance schedule continues.

The first pass stopped before any deletion because Docker changed the ordering of mount records. Its empty journal and source are retained. The validated pass canonicalizes mount ordering while retaining exact membership/state checks. Both passes are closed one-shot operations; never rerun their deletion scripts.
