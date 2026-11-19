// A room bundle produced by scripts/pipeline.mjs: splat + layout + Lux3D models + Lux3D articulation (URDF).
import {
  Box3, GLTFLoader, Object3D, Quaternion, SplatLoader, SplatUtils, Vector3, downloadTexture,
} from '@manycore/aholo-viewer';
import type { Box, Layout, Vec3 } from './layout/spatiallm';
import type { Joint, ObjectAsset } from './layout/mjcf';
import { quatFromMat } from './render/mjview';

export interface UrdfJoint { name: string; type: 'hinge' | 'slide'; parent: string; child: string; origin: number[]; axis: number[]; lower: number; upper: number }
export interface Articulated { root: string; links: Record<string, { file: string; bounds: { min: number[]; max: number[] } }>; joints: UrdfJoint[] }
export interface Manifest {
  id: string; source: string;
  splat?: string; splatFormat?: 'spz' | 'ply'; pano?: string;
  T?: number[][]; align?: Record<string, unknown>;
  layout: string;
  models: Record<string, string>;
  articulated: Record<string, Articulated>;
  timings?: Record<string, number>; counts?: Record<string, number>;
}

export async function loadManifest(id: string): Promise<{ manifest: Manifest; base: string; layoutText: string }> {
  const base = `/rooms/${id}/`;
  const manifest: Manifest = await (await fetch(base + 'manifest.json')).json();
  const layoutText = await (await fetch(base + manifest.layout)).text();
  return { manifest, base, layoutText };
}

/** glTF (Y-up) → MuJoCo/SpatialLM (Z-up): rotate +90° about X: (x, y, z) → (x, −z, y). */
const yUpToZUp = (p: number[]): Vec3 => [p[0], -p[2], p[1]];

/**
 * Which way a box's furniture faces, as quarter turns r (0..3) of the model about Z.
 * glTF front (+Z) lands on box −Y after the Y-up→Z-up turn; r rotates it to face away from the
 * nearest wall (furniture stands against walls) or toward the room's center when free-standing.
 */
export function facing(b: Box, l: Layout): number {
  let best = { d: Infinity, x: 0, y: 0 };
  for (const w of l.walls) {
    const ex = w.b[0] - w.a[0], ey = w.b[1] - w.a[1], L2 = ex * ex + ey * ey || 1;
    const t = Math.max(0, Math.min(1, ((b.center[0] - w.a[0]) * ex + (b.center[1] - w.a[1]) * ey) / L2));
    const px = w.a[0] + t * ex, py = w.a[1] + t * ey, d = Math.hypot(b.center[0] - px, b.center[1] - py);
    if (d < best.d) best = { d, x: px, y: py };
  }
  let ox = b.center[0] - best.x, oy = b.center[1] - best.y;
  if (!(best.d < 1.5)) {
    let cx = 0, cy = 0; for (const w of l.walls) { cx += w.a[0] + w.b[0]; cy += w.a[1] + w.b[1]; }
    ox = cx / (2 * l.walls.length || 1) - b.center[0]; oy = cy / (2 * l.walls.length || 1) - b.center[1];
  }
  const c = Math.cos(b.angleZ), s = Math.sin(b.angleZ);
  const lx = ox * c + oy * s, ly = -ox * s + oy * c; // outward direction in the box frame
  const dirs = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // front direction after r quarter turns
  let r = 0, bd = -Infinity;
  dirs.forEach(([dx, dy], i) => { const v = dx * lx + dy * ly; if (v > bd) { bd = v; r = i; } });
  return r;
}

/** Uniform fit of a model's bounds (glTF frame) into a box, turned r quarter turns: scale k, center c (Z-up, model units). */
function fit(bounds: { min: number[]; max: number[] }, b: Box, r: number) {
  const a = yUpToZUp(bounds.min), z = yUpToZUp(bounds.max);
  const lo = [Math.min(a[0], z[0]), Math.min(a[1], z[1]), Math.min(a[2], z[2])];
  const hi = [Math.max(a[0], z[0]), Math.max(a[1], z[1]), Math.max(a[2], z[2])];
  const size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const odd = r % 2 === 1;
  const sx = odd ? size[1] : size[0], sy = odd ? size[0] : size[1];
  const k = Math.min(b.size[0] / (sx || 1), b.size[1] / (sy || 1), b.size[2] / (size[2] || 1));
  // stand on the floor: when the model ends up shorter than its box, drop it to the box bottom
  const dz = -b.size[2] / 2 + (size[2] * k) / 2;
  return { k, c: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2] as Vec3, r, dz };
}
type Fit = ReturnType<typeof fit>;
const turn = (v: Vec3, r: number): Vec3 => { let [x, y] = v; for (let i = 0; i < r; i++) [x, y] = [-y, x]; return [x, y, v[2]]; };
const toBox = (p: number[], f: Fit): Vec3 => { const q = yUpToZUp(p); return turn([(q[0] - f.c[0]) * f.k, (q[1] - f.c[1]) * f.k, (q[2] - f.c[2]) * f.k + f.dz], f.r); };
const dirToBox = (d: number[], f: Fit): Vec3 => turn(yUpToZUp(d), f.r);

/** Assembled bounds of an articulated model (root + children at their joint origins), glTF frame. */
function assembled(art: Articulated) {
  const min = [...art.links[art.root].bounds.min], max = [...art.links[art.root].bounds.max];
  for (const j of art.joints) {
    const l = art.links[j.child]; if (!l) continue;
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], l.bounds.min[k] + j.origin[k]); max[k] = Math.max(max[k], l.bounds.max[k] + j.origin[k]); }
  }
  return { min, max };
}

export type ArticulatedAsset = ObjectAsset & { open: number[] };

/** MuJoCo joints (box frame, meters) for every articulated box, from the Lux3D URDF. */
export function articulationAssets(layout: Layout, m: Manifest): Record<string, ArticulatedAsset> {
  const out: Record<string, ArticulatedAsset> = {};
  for (const b of layout.boxes) {
    const art = m.articulated?.[b.cls];
    if (!art?.joints?.length || !art.links[art.root]) continue;
    const f = fit(assembled(art), b, facing(b, layout));
    const joints: Joint[] = [], open: number[] = [];
    for (const j of art.joints) {
      const l = art.links[j.child]; if (!l) continue;
      const lo = toBox(l.bounds.min.map((v, k) => v + j.origin[k]), f), hi = toBox(l.bounds.max.map((v, k) => v + j.origin[k]), f);
      const center: Vec3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
      const size: Vec3 = [0, 1, 2].map((k) => Math.max(0.02, Math.abs(hi[k] - lo[k]) * 0.94)) as Vec3;
      const s = j.type === 'slide' ? f.k : 1;
      const range: [number, number] = [j.lower * s, j.upper * s];
      const axis = dirToBox(j.axis, f);
      joints.push({ type: j.type, pos: toBox(j.origin, f), axis, range, part: { center, size } });
      let end = Math.abs(range[0]) > Math.abs(range[1]) ? range[0] : range[1];
      if (j.type === 'slide') {
        // drawers: never slide out more than 70% of their own depth along the axis
        const depth = Math.abs(axis[0]) * size[0] + Math.abs(axis[1]) * size[1] + Math.abs(axis[2]) * size[2];
        end = Math.sign(end) * Math.min(Math.abs(end), 0.7 * depth);
      }
      open.push(end);
    }
    out[b.id] = { joints, open };
  }
  return out;
}

export async function loadSplat(base: string, m: Manifest): Promise<Object3D | null> {
  if (!m.splat) return null;
  const type = m.splatFormat === 'ply' ? SplatLoader.SplatFileType.PLY : SplatLoader.SplatFileType.SPZ;
  const data = await SplatLoader.parseSplatData(type, new URL(base + m.splat, location.href).href, SplatLoader.SplatPackType.Compressed);
  const splat = await SplatUtils.createSplat(data);
  const holder = new Object3D();
  holder.add(splat as unknown as Object3D);
  if (m.T) {
    // T = s·R | t  (row-major) → position, quaternion, uniform scale
    const T = m.T, s = Math.hypot(T[0][0], T[1][0], T[2][0]);
    const R = [T[0][0] / s, T[0][1] / s, T[0][2] / s, T[1][0] / s, T[1][1] / s, T[1][2] / s, T[2][0] / s, T[2][1] / s, T[2][2] / s];
    const [x, y, z, w] = quatFromMat(R, 0);
    holder.position.set(T[0][3], T[1][3], T[2][3]);
    holder.quaternion = new Quaternion(x, y, z, w);
    holder.scale.set(s, s, s);
  }
  return holder;
}

async function glb(url: string, cache: Map<string, Promise<ArrayBuffer>>) {
  if (!cache.has(url)) cache.set(url, fetch(url).then((r) => r.arrayBuffer()));
  const buf = await cache.get(url)!;
  return (await GLTFLoader.loadGLTF(buf.slice(0), { textureLoader: (u: string) => downloadTexture(u) as any })).scene;
}

/** model (glTF frame) → box frame: Y-up→Z-up, scale k, offset, then optional 90° swap about Z, then position. */
function placed(model: Object3D, k: number, pos: Vec3, r: number, innerOffset: Vec3 = [0, 0, 0]) {
  model.rotation.x = Math.PI / 2;
  const inner = new Object3D(); inner.add(model); inner.scale.set(k, k, k); inner.position.set(...innerOffset);
  const outer = new Object3D(); outer.add(inner); outer.position.set(...pos);
  outer.rotation.z = (r * Math.PI) / 2;
  return outer;
}
/** world-space bounds of a loaded glTF scene in its own (Y-up) frame */
function gltfBounds(model: Object3D) {
  const bb = new Box3().setFromObject(model);
  return { min: [bb.min.x, bb.min.y, bb.min.z], max: [bb.max.x, bb.max.y, bb.max.z] };
}

/** Visuals per MuJoCo body, loaded in parallel; `onVisual` fires as each one is ready. */
export async function loadModels(base: string, m: Manifest, layout: Layout, onVisual: (body: string, obj: Object3D) => void) {
  const cache = new Map<string, Promise<ArrayBuffer>>();
  await Promise.all(layout.boxes.map(async (b) => {
    try {
      const r = facing(b, layout);
      const art = m.articulated?.[b.cls];
      if (art?.joints?.length && art.links[art.root]) {
        const f = fit(assembled(art), b, r), c = f.c;
        const parts = art.joints.filter((j) => art.links[j.child]);
        await Promise.all([
          glb(base + art.links[art.root].file, cache).then((o) => onVisual(b.id, placed(o, f.k, [0, 0, 0], f.r, [-c[0] * f.k, -c[1] * f.k, -c[2] * f.k + f.dz]))),
          ...parts.map((j, i) => glb(base + art.links[j.child].file, cache).then((o) => onVisual(`${b.id}_part${i}`, placed(o, f.k, toBox(j.origin, f), f.r)))),
        ]);
        return;
      }
      const file = m.models?.[b.cls];
      if (!file) return;
      const model = await glb(base + file, cache);
      const f = fit(gltfBounds(model), b, r), c = f.c;
      onVisual(b.id, placed(model, f.k, [0, 0, 0], f.r, [-c[0] * f.k, -c[1] * f.k, -c[2] * f.k + f.dz]));
    } catch (e) { console.warn('model', b.cls, e); }
  }));
}
