# Classical baseline (`sort_fixed`, `sort_language`)

Head RGB-D → HSV colour blobs + depth → object top centre in the pelvis frame → top-down grasp (robot / base: the
"cage", thumb against both fingers, commanded deep for the kp 1.5 fingers; sim profile: thumb-opposed pinch) planned with
Pinocchio IK → carry to the object's bin (read from the instruction text; no text: `task.fixed_mapping`, sim red → left, yellow → center, blue → right) → release → home → look again, until nothing sortable is left
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

Colours (`colors.py`): an open vocabulary. The reader accepts any colour word under a canonical name ("violet" →
purple, "amber" → orange); per episode the named colours are matched to the world's colour classes
(`perception.color.refs`, measured hues: sim red / yellow / blue, robot orange / yellow / purple): a word that is a
class uses its measured hue, another one merges with the unnamed class nearest to its lexicon hue ("red" on the robot
= the orange objects) unless that is ambiguous; colours that cannot be told apart become one class that gets a bin only
where all its names agree. The reset log's second line (`colour classes: ...`) says what was matched; each look line
carries every detection's own hue (`/H11.6`, clipped pixels excluded) to tune the references. After every look the
scene calibrates the classes (`colors.Calibrator`, `perception.color.calib`): blob hues → clusters → assigned to the
named colours one-to-one (a common white-balance shift allowed on the robot); a clear assignment moves its class for
the episode (logged `colour calibration: ...`), an unclear one decides nothing; per object a class needs a margin
(`color.obj_margin`). Tests: `test_colors.py`.

| File | What |
|---|---|
| `perception.py` | HSV masks + head depth → detections in the pelvis frame (`interface/calibration.yaml`) |
| `bins.py` | The three boxes from the levelled head depth every look (`perception.boxes`): wall points → FFT template search over yaw → per-box trimmed fit (centre, yaw, side; rim height), coverage / wall-inside checks, names by row position, fused over the episode. Mode `on` (base = robot): the detected boxes drive drop slots, carry / drop heights, rim clearances, object masks; a box not found or not certainly named gets nothing. `log` in profile sim (scene.yaml + registration as before). Tests / reports: `test_bins.py` (`python -m ...test_bins grid / reach / frames`) |
| `far.py` | Far release (`grasp.far`, on in base, off in sim): a detected box with no reachable drop slot gets a release with the hand tilted for reach (approach axis <= 60 deg, the object ahead of the wrist) up to shoulder height; the opening hand pushes the object 1.5-6 cm forward (`release_shift`, sim-measured); accepted only if it then lands inside the near wall at the nominal box pose (short = rim / counter = 0, retried), with margin at the far wall (boxes at the table's far edge: long = -1) and the sides (`release_side`: releases stay near the middle line), the wrist (camera bracket) 5 cm above the rims; landings the head camera can see preferred; else the object stays. `Planner.box_reachable`: the policy skips a box no arm can deliver into before planning (one line per look); `policy.look_budget_s` caps a look's planning. Reach map: `python -m ...far map [along ...]`. Release in motion: studied, dropped (module doc). Lab: `sim/far_lab.py`; tests `test_far.py` |
| messy objects | `perception.lying` (`_lying`): flat top / ridge (prism on a side: shape triangle, `lying`, `ridge_yaw`) / rounded top (cylinder on its side) / unsure (shape None) from the blob's height profile; `perception.touching` (`_split_touching`): one-colour blobs of touching objects cut into their objects; planner: a lying object is pinched on its two flat ends (fingers along the ridge, `grasp.face_align.lying*`, any profile), neighbours on in every profile, outer objects of a cluster first (`neighbours.order_free`). Reports: `eval_messy.py` (frames or `--mock N`); tests `test_messy.py` |
| `still.py` | Stillness gate before a look (`still`, robot profile): two head images `dt_s` apart, corners outside the arms' silhouettes tracked + RANSAC similarity, look when the motion <= `thr_px`, else wait (timeout: look anyway, logged) |
| `kinematics.py` | Pinocchio FK/IK on the G1 URDF (`third_party/xr_teleoperate/.../g1_body29_hand14.urdf`) |
| `planner.py` | Cartesian waypoints → 30 Hz joint trajectories, safety pre-check; paired cycle (`two_arm`, sim profile): one object per arm from one look, merged with the second arm delayed until the arm-arm sphere model keeps `min_clearance`; parked arm (`two_arm.avoid_parked`, sim profile): a single-arm plan that comes too close to the waiting arm gets an evade (that arm moves to a `two_arm.clear_poses` pose and back in the same queue) or is skipped; drop slots (`grasp.slots`): reach-tested release points per bin and arm, the one farthest from what is already in the bin; neighbours (`neighbours`, sim profile): every tick keeps the arm's sphere model clear of the other objects (cylinders); face-aligned grasps (`grasp.face_align`, sim profile) from the detected shape + yaw |
| `instruction.py` | Instruction text → mapping + evidence + problems (`parse_instruction`); one object's bin (`object_bin`) |
| `colors.py` | The episode's colour classes: the mapping's colour words matched to the world's measured classes (`match_colors`) |
| `policy.py` | `ClassicalPolicy`: state machine; target bin per object from the instruction (`bin_for`); plans in a thread and returns holds meanwhile; per-object tries; delivered objects' release points (slots); an object blocked by a neighbour waits for it, least-bad rule last |
| `verify.py` | Grasp verification from the finger joints (`verify`, sim profile): a verified miss / slip aborts the carry |
| settle and correct | `grasp.correct` (on, base and sim): the arm dwells at the grasp pose (phase `settle`), the policy measures the grasp point's tracking error (FK of measured vs commanded joints: the PD arms sag ~8 mm without gravity compensation) and shifts that arm's queued ticks to cancel it before the hand closes (`Planner.correction_dq` / `correct`, every guard or the queue stays as planned; log line `correct <side>: ...`); tests `test_correct.py` |
| `sag.py` | Gravity-sag feed-forward (`sag`, on in base and sim): the arms are position-controlled without gravity compensation (sim ~300 Nm/rad; the robot runner kp 80 / 40, no torque feed-forward: 3.5-10x the sim's sag, 2-5 cm at the hand), so the policy sends intended + clip(-G * gravity torque) and plans / checks / corrects in intended space (the planner's dry run and speed limits see the commanded ticks). G is learned online from still samples (the detector's probe, home settles, grasp settles; robust per-joint least squares around the prior, bounded, refuses contact samples, compensates less where the error is not proportional to the torque) and blended in only in the settle before a look. Log lines `sag feed-forward: ...`; replay on the sim recordings: grasp point 9 mm -> 0.5 mm before the correction; tests `test_sag.py` |
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
for, settle and correct off, sag feed-forward off; also for sim A/B runs), `TestSimProfile*`, `test_correct.py` and
`test_sag.py` on `sim` / `none`.

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
