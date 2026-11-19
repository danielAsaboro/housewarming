// Layout (SpatialLM, Z-up meters) → MuJoCo MJCF. MuJoCo is also Z-up meters, so no frame change here.
import type { Box, Layout, Opening, Vec3, Wall } from './spatiallm';

export interface Joint {
  type: 'hinge' | 'slide';
  /** joint anchor in the object's box frame (meters, origin at box center) */
  pos: Vec3;
  /** unit axis in the object's box frame */
  axis: Vec3;
  /** radians for hinge, meters for slide */
  range: [number, number];
  /** the moving part, as a box in the object's box frame */
  part: { center: Vec3; size: Vec3 };
  mesh?: string;
}

export interface ObjectAsset {
  /** mesh file name inside the MuJoCo VFS, already scaled to the box */
  mesh?: string;
  joints?: Joint[];
}

export interface SceneOptions {
  wallThickness?: number;
  objects?: Record<string, ObjectAsset>;
  /** MJCF file to <include>, e.g. a Menagerie robot; its free joint is placed via spawn() */
  robotInclude?: string;
}

const NON_COLLIDING = new Set(['carpet', 'curtain', 'painting', 'chandelier', 'illumination', 'wall decoration', 'plants']);
const MOVABLE: Record<string, number> = {
  chair: 6, 'dining chair': 6, stool: 3, 'side table': 8, 'coffee table': 15, nightstand: 12, 'trash can': 2, ottoman: 5,
};

export function isMovable(cls: string) { return cls in MOVABLE; }
export function collides(cls: string) { return !NON_COLLIDING.has(cls); }

export function buildScene(layout: Layout, opts: SceneOptions = {}): string {
  const t = opts.wallThickness ?? 0.08;
  const floorZ = layout.walls.length ? median(layout.walls.map((w) => Math.min(w.a[2], w.b[2]))) : 0;
  const centroid = roomCentroid(layout.walls);
  const meshes = new Set<string>();
  const L: string[] = [];

  L.push('<mujoco model="housewarming">');
  L.push('  <compiler angle="radian" autolimits="true"/>');
  // Lite3's policy was deployed at 1 kHz physics (Lite3_rl_deploy mujoco_simulation.py DT=0.001)
  L.push(`  <option timestep="${opts.robotInclude ? 0.001 : 0.002}" integrator="implicitfast"/>`);
  if (opts.robotInclude) L.push(`  <include file="${esc(opts.robotInclude)}"/>`);
  L.push('  <default>');
  L.push('    <default class="wall"><geom type="box" rgba="0.92 0.9 0.86 1" friction="0.8"/></default>');
  L.push('    <default class="static"><geom type="box" rgba="0.6 0.55 0.5 1" friction="0.8"/></default>');
  L.push('    <default class="ghost"><geom type="box" contype="0" conaffinity="0" rgba="0.5 0.5 0.5 0.3"/></default>');
  L.push('  </default>');

  const assets: string[] = [];
  for (const o of Object.values(opts.objects ?? {})) {
    for (const m of [o.mesh, ...(o.joints ?? []).map((j) => j.mesh)]) if (m && !meshes.has(m)) { meshes.add(m); assets.push(`    <mesh name="${meshName(m)}" file="${esc(m)}"/>`); }
  }
  if (assets.length) L.push('  <asset>', ...assets, '  </asset>');

  L.push('  <worldbody>');
  L.push('    <light pos="0 0 4" dir="0 0 -1" directional="true"/>');
  L.push(`    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 ${f(floorZ)}" rgba="0.8 0.78 0.74 1" friction="1"/>`);

  for (const w of layout.walls) {
    const openings = layout.openings.filter((o) => o.wallId === w.id);
    for (const [i, g] of wallPieces(w, openings, t, centroid, floorZ).entries()) {
      L.push(`    <geom name="${esc(w.id)}_${i}" class="wall" pos="${v(g.pos)}" euler="0 0 ${f(g.yaw)}" size="${v(g.half)}"/>`);
    }
  }

  for (const b of layout.boxes) L.push(...objectBody(b, opts.objects?.[b.id]));
  L.push('  </worldbody>');
  // soft position servos so doors/drawers can be opened on command and still be pushed by the robot
  const acts: string[] = [];
  for (const b of layout.boxes) for (const [i] of (opts.objects?.[b.id]?.joints ?? []).entries()) {
    const j = `${esc(b.id)}_j${i}`;
    acts.push(`    <position name="${j}_act" joint="${j}" kp="6" kv="1.5" ctrlrange="-10 10"/>`);
  }
  if (acts.length) L.push('  <actuator>', ...acts, '  </actuator>');
  L.push('</mujoco>');
  return L.join('\n');
}

function objectBody(b: Box, asset?: ObjectAsset): string[] {
  const half: Vec3 = [b.size[0] / 2, b.size[1] / 2, b.size[2] / 2];
  const name = esc(b.id);
  const out = [`    <body name="${name}" pos="${v(b.center)}" euler="0 0 ${f(b.angleZ)}">`];
  const solid = collides(b.cls);
  if (solid && isMovable(b.cls) && !asset?.joints?.length) out.push(`      <freejoint name="${name}_free"/>`);
  const cls = solid ? 'static' : 'ghost';
  const mass = isMovable(b.cls) ? ` mass="${MOVABLE[b.cls]}"` : '';
  if (asset?.mesh) {
    out.push(`      <geom name="${name}_vis" type="mesh" mesh="${meshName(asset.mesh)}" contype="0" conaffinity="0" group="1"/>`);
    out.push(`      <geom name="${name}_col" class="${cls}" size="${v(half)}" group="3"${mass}/>`);
  } else {
    out.push(`      <geom name="${name}_col" class="${cls}" size="${v(half)}"${mass}/>`);
  }
  for (const [i, j] of (asset?.joints ?? []).entries()) {
    const ph: Vec3 = [j.part.size[0] / 2, j.part.size[1] / 2, j.part.size[2] / 2];
    out.push(`      <body name="${name}_part${i}">`);
    out.push(`        <joint name="${name}_j${i}" type="${j.type}" pos="${v(j.pos)}" axis="${v(j.axis)}" range="${f(j.range[0])} ${f(j.range[1])}" damping="0.5" armature="0.01"/>`);
    // parts are driven by their position servo; they don't collide (they'd catch the wall the cabinet stands against)
    out.push(`        <geom name="${name}_part${i}_col" type="box" pos="${v(j.part.center)}" size="${v(ph)}" mass="1" contype="0" conaffinity="0" rgba="0.55 0.45 0.35 1"/>`);
    if (j.mesh) out.push(`        <geom type="mesh" mesh="${meshName(j.mesh)}" contype="0" conaffinity="0" group="1"/>`);
    out.push('      </body>');
  }
  out.push('    </body>');
  return out;
}

interface Piece { pos: Vec3; half: Vec3; yaw: number }

/** Split a wall into solid boxes around its door/window openings. Exported for tests. */
export function wallPieces(w: Wall, openings: Opening[], t: number, centroid: [number, number], floorZ: number): Piece[] {
  const dx = w.b[0] - w.a[0], dy = w.b[1] - w.a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-3 || w.height <= 0) return [];
  const ux = dx / len, uy = dy / len;
  // SpatialLM walls are inner contour edges: push the slab outward, away from the room centroid.
  let nx = -uy, ny = ux;
  const mx = (w.a[0] + w.b[0]) / 2, my = (w.a[1] + w.b[1]) / 2;
  if ((centroid[0] - mx) * nx + (centroid[1] - my) * ny > 0) { nx = -nx; ny = -ny; }
  const off = t / 2;
  const yaw = Math.atan2(uy, ux);
  const H = w.height;
  const base = Math.min(w.a[2], w.b[2], floorZ);

  const holes = openings
    .map((o) => {
      const s = (o.center[0] - w.a[0]) * ux + (o.center[1] - w.a[1]) * uy;
      const z = o.center[2] - base;
      return { s0: clamp(s - o.width / 2, 0, len), s1: clamp(s + o.width / 2, 0, len), z0: clamp(z - o.height / 2, 0, H), z1: clamp(z + o.height / 2, 0, H) };
    })
    .filter((h) => h.s1 - h.s0 > 0.05)
    .sort((a, b) => a.s0 - b.s0);

  const pieces: Piece[] = [];
  const add = (s0: number, s1: number, z0: number, z1: number) => {
    if (s1 - s0 < 0.02 || z1 - z0 < 0.02) return;
    const s = (s0 + s1) / 2, z = (z0 + z1) / 2;
    pieces.push({
      pos: [w.a[0] + ux * s + nx * off, w.a[1] + uy * s + ny * off, base + z],
      half: [(s1 - s0) / 2, t / 2, (z1 - z0) / 2],
      yaw,
    });
  };
  let cursor = 0;
  for (const h of holes) {
    add(cursor, h.s0, 0, H);
    const a = Math.max(cursor, h.s0);
    add(a, h.s1, 0, h.z0);
    add(a, h.s1, h.z1, H);
    cursor = Math.max(cursor, h.s1);
  }
  add(cursor, len, 0, H);
  return pieces;
}

function roomCentroid(walls: Wall[]): [number, number] {
  if (!walls.length) return [0, 0];
  let x = 0, y = 0;
  for (const w of walls) { x += w.a[0] + w.b[0]; y += w.a[1] + w.b[1]; }
  return [x / (2 * walls.length), y / (2 * walls.length)];
}

const median = (a: number[]) => { const s = [...a].sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const f = (x: number) => (Math.abs(x) < 1e-9 ? '0' : Number(x.toFixed(4)).toString());
const v = (a: readonly number[]) => a.map(f).join(' ');
const esc = (s: string) => s.replace(/[<>&"]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' })[c]!);
const meshName = (file: string) => esc(file.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_]/g, '_'));
