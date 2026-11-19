# Third-party code and method provenance

## Minimal vendored runtime

The files under `src/lewm_proposals/vendor/` are derived from the MIT-licensed
LeWM / stable-worldmodel implementation inspected in:

`https://github.com/Thyguad/lewm-research/tree/67cf53f26b96033b53b37eb04c89b82c3c88b933`

| Local file | Source path in that revision | Local changes |
|---|---|---|
| `lewm.py` | `stable-worldmodel/stable_worldmodel/wm/lewm/lewm.py` | Source/license comment only |
| `module.py` | `stable-worldmodel/stable_worldmodel/wm/lewm/module.py` | Source/license comment only |
| `cube_env.py` | `stable-worldmodel/stable_worldmodel/envs/ogbench/cube_env.py` | Two imports redirected to local `spaces` and `camera` |
| `spaces.py` | `stable-worldmodel/stable_worldmodel/spaces.py` | Replace package-wide `swm.utils.get_in` dependency with equivalent dictionary traversal |
| `camera.py` | `perturb_camera_angle` in `stable-worldmodel/stable_worldmodel/envs/utils.py` | Extract that function with NumPy import; omit unrelated PushT rendering dependencies |

Original upstream: https://github.com/galilai-group/stable-worldmodel .
The MIT license is retained in `src/lewm_proposals/vendor/LICENSE`.
The source snapshot is recorded for reproducibility; the new project does not import,
clone or require the previous research working directory at runtime.

`frozen.py` constructs Hugging Face's ViT-tiny using the same dimensions as
`stable_pretraining.backbone.utils.vit_hf`: hidden size 192, 12 layers, 3 heads,
intermediate size 768, no pooling layer. Patch/image sizes and mask-token setting
come from the official checkpoint config. This avoids importing the full pretraining
framework for inference; strict state-dict loading and the optional real-weight parity
test check compatibility. The factory was inspected in the official
https://github.com/galilai-group/stable-pretraining implementation.

Cube reset, action semantics and CEM final-mean execution were inspected in
`le-wm/cube_research/{data,model,simulator,protocol}.py` at the same source revision.
The new evaluator does not include that project's adaptation or perturbation experiments.

OGBench, MuJoCo, dm-control, PyTorch, Transformers and other installed dependencies
retain their own licenses. OGBench assets are provided by its installed package,
not copied into this repository. Official datasets and pretrained weights are separate
downloads and are never committed here.

## Related work

- Maes et al., *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*, https://arxiv.org/abs/2603.19312 .
- Kim and Sreenath, *Joint World Action Models through Differentiable Trajectory Optimization*, https://arxiv.org/abs/2611.17626 .

The latter supplies a close joint state/action flow baseline and a model-size reference.
This repository implements stage-one conditional flow matching plus optional frozen-model
candidate ranking; it does **not** implement WATO's differentiable trajectory-optimization
training. Do not label its results as a WATO reproduction or claim this baseline composition
as a novel method.
