# Housewarming
One line: a phone video of your room becomes a physics scene a robot can walk, open and bump into.
For / Job / Done: robotics devs and hobbyists · a sim-ready copy of a real room with no hand-modelling · upload a room video → browser MuJoCo scene with walls, true-size furniture, opening doors/drawers, a quadruped; MJCF zip export.

Sponsor mechanism (critical path, Aholo API, base https://api.aholo.ai/global, raw `Authorization: <key>`, no Bearer):
1. Reconstruction (World `/world/v1/reconstructions` or Reality `/web-scan/v2/...` for metric scale + world_up) → PLY/SPZ splat
2. Spatial Understanding `/spatial-understanding/v1/tasks` (SpatialLM) → text lines Wall/Door/Window/Bbox in meters, Z-up
3. Lux3D img→3D / text→3D (G1, aiPredictSize false) per Bbox, scaled to the box
4. Lux3D articulation `/lux3d/v1/articulation-animation/task/create` (glbUrl + motion prompt) → joints
5. @manycore/aholo-viewer renders splat + MuJoCo bodies
Spec: ../hackathon/sponsor-docs/openapi.json (Chinese descriptions) and llms.txt.

Features in scope (5): room in · room structure → colliders · objects + articulation · browser sim with quadruped · MJCF export.
Not building: policy training, sim-to-real claims, Isaac/USD export, multi-room, mobile capture app, accounts, robot arm.

Stack: Vite + TypeScript, @mujoco/mujoco 4.0.1 (official WASM), @manycore/aholo-viewer 1.14 (MIT), vitest. Node scripts for the Aholo pipeline in scripts/.
Run: `pnpm dev` · Test: `pnpm test` · Spike: `pnpm spike` (needs .env) · Build: `pnpm build`
Repo map: src/ app · src/layout/ SpatialLM parser + MJCF builder · scripts/ Aholo pipeline + spike · tests/ · public/ assets.

Invariants (never):
- never present a fixture, hand-written layout or placeholder mesh as Aholo output; fixtures live in tests/ only
- never commit .env or API responses containing signed URLs
- MuJoCo is Z-up meters; SpatialLM is Z-up meters; splats are often OpenCV (Y-down) — convert in one place (src/layout/frames.ts)
- free every MuJoCo WASM object with .delete()
- register DOM events on the viewer container, never on the engine canvas

Design: follow design.md (Dribbble references first; house keys / blueprint concept). No taglines, helper text or secondary labels.
Evidence: a claim is done only with a row in ../hackathon/evidence.md (real response, file, or URL). Mocks are never evidence.
Forbidden artifacts: notes, research, plans live in ../hackathon — never in this repo.
Commits: no Claude co-author or session trailers.
