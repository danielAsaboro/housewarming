# Housewarming

Film your room, get a physics simulation a robot can walk, bump into and open.

**Live:** https://housewarming-lime-one.vercel.app · Manycore Aholo Global 3D AI Hackathon (#Manycorehackathon)

## How it works

```
phone video / Insta360 / photo
  └─ Aholo World            reconstruction or generation → 3D Gaussian splat (PLY/SPZ)
       └─ align             up axis, floor at z=0, walls on X/Y, meters   (scripts/lib/align.mjs)
            └─ Aholo Spatial Understanding (SpatialLM)  walls · doors · windows · object boxes
                 ├─ MJCF    wall slabs with door/window cut-outs, fixed + pushable furniture (src/layout/mjcf.ts)
                 ├─ Aholo Lux3D text→3D   one true-size model per furniture class
                 └─ Aholo Lux3D articulation   animated GLB → hinge / slide joints (scripts/lib/joints.mjs)
                      └─ browser: MuJoCo 4 WASM + Aholo Viewer + DEEP Robotics Lite3 on its own PPO policy
```

## Run

```bash
pnpm install
pnpm dev            # http://localhost:5710 — opens the first room in public/rooms/index.json
pnpm test           # parser, scene builder, alignment, Lite3 walking (real MuJoCo + ONNX in Node)
```

Controls: `W/S` walk · `A/D` turn · `Q/E` sidestep · `F` open/close the nearest door or drawer · `V` room ↔ blueprint · `M` download MJCF · `R` reset · drag to orbit, wheel to zoom.

## Build a room

```bash
cp .env.example .env   # AHOLO_API_KEY from labs.aholo.ai → API keys
node --env-file=.env scripts/pipeline.mjs --id my-kitchen --video ./kitchen.mp4      # metric reconstruction
node --env-file=.env scripts/pipeline.mjs --id loft --image ./loft.jpg               # world from one photo
node --env-file=.env scripts/pipeline.mjs --id demo --prompt "a small kitchen with…" # world from text
```

Then add the id to `public/rooms/index.json` (or open `/?room=<id>`). Each step is resumable; state lives in `public/rooms/<id>/state.json`.

## Robot

DEEP Robotics Lite3 MJCF (`deep_robotics_model`, BSD-3) and walking policy (`Lite3_rl_deploy/policy/ppo/policy.onnx`, BSD-3), vendored by `scripts/vendor-lite3.sh` at pinned commits. `src/robot/lite3.ts` ports their ONNX runner: PD at 1 kHz (kp 30, kd 1), policy every 12 steps, 45-value observation.

## Credits

MuJoCo 4.0.1 WASM (Google DeepMind, Apache-2.0) · @manycore/aholo-viewer (MIT) · Aholo SDKs (MIT) · DEEP Robotics Lite3 (BSD-3) · ONNX Runtime Web (MIT) · fflate (MIT).
