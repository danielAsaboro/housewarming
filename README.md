# Housewarming

Film your room, get a physics simulation a robot can walk, bump into and open.

**Live:** https://housewarming-lime-one.vercel.app · Manycore Aholo Global 3D AI Hackathon (#Manycorehackathon)

## How it works

```
phone video · Insta360 · photo · text
  └─ Aholo World                 reconstruction or generation → 3D Gaussian splat (PLY + SPZ)
       └─ align                  up axis, floor at z=0, walls on X/Y, meters            scripts/lib/align.mjs
            ├─ occupancy         10 cm grid of what stands between knee and head height  scripts/lib/occupancy.mjs
            └─ Aholo Spatial Understanding (SpatialLM)   walls · doors · windows · object boxes
                 ├─ MJCF         wall slabs with door/window cut-outs; heavy furniture fixed, chairs pushable   src/layout/mjcf.ts
                 ├─ Aholo Lux3D text→3D        one metric model per furniture class, fitted to its box, facing away from its wall
                 └─ Aholo Lux3D articulation   URDF (revolute + prismatic) → MuJoCo hinge/slide joints with soft servos
                      └─ browser: MuJoCo 4 WASM physics · Aholo Viewer rendering · DEEP Robotics Lite3 on its own PPO policy
```

What you see:
- **Room** — the real scan, with the simulation's blueprint traced over it. Doors and drawers show their Lux3D model while open.
- **Blueprint** — the simulation itself: wall slabs, colliders, every Lux3D model, the robot.
- On load, a build reveal replays the pipeline: room → walls → detected objects → Lux3D models → robot.

## Run

```bash
pnpm install
pnpm dev            # http://localhost:5710
pnpm studio         # optional: lets the app build new rooms from an uploaded video/photo (needs .env)
pnpm test           # parser, scene builder, alignment, facing, joints, Lite3 walking (real MuJoCo + ONNX in Node)
```

Controls: `W/S` walk · `A/D` turn · `Q/E` sidestep · `F` open/close what's in reach · `V` room ↔ blueprint · `M` download MJCF · `R` replay the build · drag to orbit · wheel to zoom.

## Build a room

```bash
cp .env.example .env   # AHOLO_API_KEY from labs.aholo.ai → API keys
node --env-file=.env scripts/pipeline.mjs --id my-kitchen --video ./kitchen.mp4
node --env-file=.env scripts/pipeline.mjs --id loft --image ./loft.jpg --metric
node --env-file=.env scripts/pipeline.mjs --id demo --prompt "a small kitchen with…" --metric
```

Or run `pnpm studio` and use **New room** in the app's room menu. Each step is resumable (`public/rooms/<id>/state.json`); add the id to `public/rooms/index.json` to list it. Deploy with `scripts/deploy.sh` (strips working files).

Measured on the kitchen: world 12 min · Spatial Understanding 55 s (26 walls, 8 doors, 13 objects) · six Lux3D models ~7 min in parallel (Aholo allows 5 concurrent tasks; the pipeline queues) · articulation 3–4 min per object.

## Robot

DEEP Robotics Lite3 MJCF (`deep_robotics_model`, BSD-3) and walking policy (`Lite3_rl_deploy/policy/ppo/policy.onnx`, BSD-3), vendored by `scripts/vendor-lite3.sh` at pinned commits. `src/robot/lite3.ts` ports their ONNX runner: PD at 1 kHz (kp 30, kd 1), policy every 12 steps, 45-value observation. Measured: 0.84 m/s on a 0.8 m/s command, 0.71 rad/s turning, stops at walls; falls are recovered automatically.

## Known limits

- Spatial Understanding can miss objects (the kitchen island and counters); physics has no collider there. The camera still avoids them using the scan's occupancy grid.
- Phone-video reconstructions are scaled to a 2.7 m ceiling unless `--metric` is passed.
- MJCF export ships colliders + robot; Lux3D GLBs are not converted for desktop MuJoCo.

## Credits

MuJoCo 4.0.1 WASM (Google DeepMind, Apache-2.0) · @manycore/aholo-viewer (MIT) · Aholo SDKs (MIT) · DEEP Robotics Lite3 (BSD-3) · ONNX Runtime Web (MIT) · glTF-Transform (MIT) · fflate (MIT).
