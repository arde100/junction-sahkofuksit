# Classical baseline (`sort_fixed`, `sort_language`)

Head RGB-D → HSV colour blobs + depth → object top centre in the pelvis frame → top-down "claw" grasp planned with
Pinocchio IK → carry to the object's bin (read from the instruction text; rung 1: red → left, yellow → center, blue → right) → release → home → look again, until nothing sortable is left
(`done`). Plans are pre-checked against the safety filter (0 clips). Cross-body objects are put down where the other
arm can finish. Meant as a floor and a code reference.

Instruction (`instruction.py`, rung 2): a rule-based reader (pure Python, no model, < 1 ms) turns the sentence into a
`scene.resolve_bin` mapping (keys colour, shape, (colour, shape), `default`) and reports per key the bin, the clause
it came from and a confidence, plus problems (conflicts, negation, bins without objects, unknown colours, ...). Every
training-template sentence parses exactly; paraphrases (either order, no "bin" word, synonyms, lists, shape and mixed
instructions, "everything else") are read by clause segmentation and pairing, not by patterns. "box(es)" = bins,
"block(s)" = any object. An object gets a bin only if every colour / shape it may have (shape not recognised, colour
unknown) leads to the same bin; else, or if its key was not understood or conflicts, it stays on the counter (0, not
a -1 guess). Nothing understood -> nothing sorted; no instruction -> the fixed mapping. The reset log line says what
was understood. Tests: `test_instruction.py` (templates, 128 development + held-back paraphrases, refusals),
`test_classical.py TestInstructionEpisodes` (mock episodes).

| File | What |
|---|---|
| `perception.py` | HSV masks + head depth → detections in the pelvis frame (`interface/calibration.yaml`) |
| `kinematics.py` | Pinocchio FK/IK on the G1 URDF (`third_party/xr_teleoperate/.../g1_body29_hand14.urdf`) |
| `planner.py` | Cartesian waypoints → 30 Hz joint trajectories, safety pre-check; paired cycle (`two_arm`, sim profile): one object per arm from one look, merged with the second arm delayed until the arm-arm sphere model keeps `min_clearance`; parked arm (`two_arm.avoid_parked`, sim profile): a single-arm plan that comes too close to the waiting arm gets an evade (that arm moves to a `two_arm.clear_poses` pose and back in the same queue) or is skipped; drop slots (`grasp.slots`): reach-tested release points per bin and arm, the one farthest from what is already in the bin; neighbours (`neighbours`, sim profile): every tick keeps the arm's sphere model clear of the other objects (cylinders); face-aligned grasps (`grasp.face_align`, sim profile) from the detected shape + yaw |
| `instruction.py` | Instruction text → mapping + evidence + problems (`parse_instruction`); one object's bin (`object_bin`) |
| `policy.py` | `ClassicalPolicy`: state machine; target bin per object from the instruction (`bin_for`); plans in a thread and returns holds meanwhile; per-object tries; delivered objects' release points (slots); an object blocked by a neighbour waits for it, least-bad rule last |
| `verify.py` | Grasp verification from the finger joints (`verify`, sim profile): a verified miss / slip aborts the carry |
| settle and correct | `grasp.correct` (on, base and sim): the arm dwells at the grasp pose (phase `settle`), the policy measures the grasp point's tracking error (FK of measured vs commanded joints: the PD arms sag ~8 mm without gravity compensation) and shifts that arm's queued ticks to cancel it before the hand closes (`Planner.correction_dq` / `correct`, every guard or the queue stays as planned; log line `correct <side>: ...`); tests `test_correct.py` |
| `policy_server.py` | The same policy behind the submission protocol (port 8000) = **reference submission** |
| `autoprofile.py` | One image for the sim selection and the robot finals (`AALTO_PROFILE=auto`, the image default): at the start of every episode a 0.6 s probe (arms held where measured, fingers curled 0.2 rad) tells the Isaac sim from the real G1 by two signatures, finger error at rest (sim < 1 mrad, real >= 48 mrad) and arm error vs the sim's sag model (sim ~0.1 mrad, real >= 8 mrad); the episode then runs on profile `sim` or `none` (both built at start-up). When in doubt: `sim`. One log line per episode (`autoprofile ep N: ...`); tests `test_autoprofile.py` |
| `config.yaml` | Everything tunable (HSV ranges, grasp, home pose, speeds, hand postures); `profiles.sim` = sim-tuned overrides |
| `run.py`, `sim_eval.py`, `mock_world.py` | In-process runs: mock world / real robot, Isaac sim scoring, rendered mock |

## Numbers (2026-11-11, nominal sim = DR off, rung 1)

| Where | Result |
|---|---|
| Aalto Isaac sim, `sim_eval` in-process, seeds 0–7 | **31/38 objects (82 %)**, 2/8 episodes all-correct, 0 clips; misses: 6 triangles (pinch slips/spins), 1 cube; 2 episodes hit the 120 s limit |
| Aalto Isaac sim, image `aalto/baseline-classical:v1` through `sim.evaluate --container --gpus none`, seeds 0–7 | **29/38 (76 %)**, 0 clips, command rate 30.0 Hz, sim gate PASS |
| Before (same day, old config) | 0/24 through the runner (seeds 0–4): perception 15–20 mm short, grasps slipped |
| Rendered mock world, seed 1 | 4/4, 0 clips |
| Real robot | not run yet (M8) |

Sim runs had 2–3 Isaac instances on twinflow at once (the sim then runs at ~0.65x real time, which makes the
30 Hz motions look ~1.5x faster to the physics). Expect +-10 % between runs.

What fixed it (2026-11-11): (1) HSV saturation floor 120 -> 70: lit top faces are washed out, so only the front face
was segmented and the "top" landed on its upper edge, 15-20 mm short; now ~4 mm. (2) Grasp point shifted 15 mm
sideways in the palm (thumb vs one finger; centred, the object wedged between index and middle). (3) `open` = the
preshape. (4) Profile `sim` (`config.yaml` bottom): sim home pose = the task's `ARM_HOME`, faster motion (0.97 rad/s),
looser IK/clearances, shorter holds. (5) 6-tick hold chunks while planning + a low-priority planner thread, so the
command rate stays at 30 Hz under CPU load.

**Sim-tuned (profile `sim`, the default; `AALTO_PROFILE=none` or `run.py --backend real` = robot base values):**
home pose, speeds/accelerations, holds, counter clearance 2 cm, IK tilt 35 deg / joint step 0.6, and the oracle's
planner features (joint-space carry first, 1 cm segment shortcut, no bin margin, cached arm-vs-bin reach test),
face-aligned grasps (shape + yaw from perception, `shape_conf >= 0.6`) and neighbour-aware plans (5 mm sphere-model
margin). Drop slots (`grasp.slots`) are on in the base config too: they are derived from `scene.yaml` + the configured
reach (switch off -> `grasp.drop_offsets` as before). The HSV floor,
grasp point and `open` posture are in the base config but were also only checked in the sim.
Tests of these: `test_grasp_plan.py`. Unit tests run on profile `sim_legacy` (`sim` + the grasp they were written
for, settle and correct off; also for sim A/B runs), `TestSimProfile*` and `test_correct.py` on `sim`.

## Run it

```bash
# in-process (conda env with pinocchio, opencv, scipy)
python -m aaltonomous.baselines.classical.run --backend mock --seed 3          # rendered mock world, scored
python -m aaltonomous.baselines.classical.run --backend mock --dry-run         # detections + plans, no motion
python -m aaltonomous.baselines.classical.sim_eval --seeds 0 1 2 3 4 5 6 7    # Isaac sim running (2nd instance: AALTO_* env vars)
python -m unittest aaltonomous/baselines/classical/test_classical.py

# as a policy server / submission image (CPU only, ~1.5 GB, up in < 2 s)
python -m aaltonomous.baselines.classical.policy_server                        # :8000 (PORT env to change)
docker build -f aaltonomous/docker/baseline.Dockerfile -t aalto/baseline-classical:v1 .
python -m aaltonomous.deploy.replay --synthetic 100 --container aalto/baseline-classical:v1 --gpus none
python -m aaltonomous.deploy.runner --backend sim --container aalto/baseline-classical:v1 --team baseline --episodes 5
```

Real robot (organisers only, robot on the gantry, operator confirms every object): `python -m
aaltonomous.baselines.classical.run --backend real --network-interface <iface>` (see `run.py --help`).

Server env vars: `AALTO_TIME_LIMIT` (s, default 120; the policy skips an object it can't finish in time),
`AALTO_ARM_VEL` (overrides `motion.arm_max_vel`), `AALTO_PROFILE` (config profile; unset = the file's `sim`; `auto` =
detect sim / robot per episode, the image's default; any explicit profile, e.g. `docker run -e AALTO_PROFILE=sim`,
forces it without detection). Calibration and `config.yaml` are baked into the image: rebuild after changing them.
Mock worlds that imitate either world's joint signatures: `make_mock_env(world="sim" | "real")` (`mock_world.py`
`SIM_WORLD` / `REAL_WORLD`: finger no-load error, thumb_0 bias, lag, the runner's arm gains without feed-forward).
