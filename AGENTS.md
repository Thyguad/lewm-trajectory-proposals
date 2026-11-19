# Project scope

Implement and evaluate stage-one joint trajectory proposals with a frozen LeWM encoder and predictor on Cube. Read README.md and docs/PROTOCOL.md before changing scientific behavior.

- The only trainable network is `JointFlow`; the predictor is not in the training graph.
- Keep `cem`, `flow_direct`, and `flow_ranked` under the same initial-state, target and MPC protocol.
- Current/goal RGB embeddings are policy inputs. Simulator state, object position, rewards and success flags belong only to the evaluator and must not enter policy decisions.
- Preserve the 5D raw action → five-action block → statistical normalization order. Score the exact clipped commands that execute.
- Split whole source episodes before sampling windows. Never silently regenerate or overwrite split files or mix test episodes into training.
- Preserve frozen checkpoint preprocessing. G has train-only latent scaling; predictor scoring remains in original LeWM latent units.
- Do not add self-evolution, old perturbation campaigns, prefix-value heads or multi-scale claims without a new user decision.
- Keep datasets, credentials, weights, caches and runs out of Git. Inputs are read-only; all downloads are explicit.
- Restore training only with matching inputs/configuration. Retain raw measurements and mark incomplete runs.
- Run relevant tests; report CPU, real-weight, simulator and GPU evidence separately. Engineering checks are not research results.
- User authorization to implement does not authorize a full GPU campaign or rental spend. Prepare concrete commands and resource measurements first.

Minimal dependencies are vendored with notices. Do not replace the Cube environment or checkpoint architecture without parity checks and an explicit protocol change.
