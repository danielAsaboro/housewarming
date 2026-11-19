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

/** Uniform fit of a model's bounds (glTF frame) into a box: scale k, center c (Z-up, model units), optional 90° swap. */
function fit(bounds: { min: number[]; max: number[] }, b: Box) {
  const a = yUpToZUp(bounds.min), z = yUpToZUp(bounds.max);
  const lo = [Math.min(a[0], z[0]), Math.min(a[1], z[1]), Math.min(a[2], z[2])];
  const hi = [Math.max(a[0], z[0]), Math.max(a[1], z[1]), Math.max(a[2], z[2])];
  const size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const swap = (size[0] > size[1]) !== (b.size[0] > b.size[1]);
  const sx = swap ? size[1] : size[0], sy = swap ? size[0] : size[1];
  const k = Math.min(b.size[0] / (sx || 1), b.size[1] / (sy || 1), b.size[2] / (size[2] || 1));
  return { k, c: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2] as Vec3, swap };
}
type Fit = ReturnType<typeof fit>;
const swapZ = (v: Vec3, f: Fit): Vec3 => (f.swap ? [-v[1], v[0], v[2]] : v);
const toBox = (p: number[], f: Fit): Vec3 => { const q = yUpToZUp(p); return swapZ([(q[0] - f.c[0]) * f.k, (q[1] - f.c[1]) * f.k, (q[2] - f.c[2]) * f.k], f); };
const dirToBox = (d: number[], f: Fit): Vec3 => swapZ(yUpToZUp(d), f);

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
    const f = fit(assembled(art), b);
    const joints: Joint[] = [], open: number[] = [];
    for (const j of art.joints) {
      const l = art.links[j.child]; if (!l) continue;
      const lo = toBox(l.bounds.min.map((v, k) => v + j.origin[k]), f), hi = toBox(l.bounds.max.map((v, k) => v + j.origin[k]), f);
      const center: Vec3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
      const size: Vec3 = [0, 1, 2].map((k) => Math.max(0.02, Math.abs(hi[k] - lo[k]) * 0.94)) as Vec3;
      const s = j.type === 'slide' ? f.k : 1;
      const range: [number, number] = [j.lower * s, j.upper * s];
      joints.push({ type: j.type, pos: toBox(j.origin, f), axis: dirToBox(j.axis, f), range, part: { center, size } });
      open.push(Math.abs(range[0]) > Math.abs(range[1]) ? range[0] : range[1]);
    }
    out[b.id] = { joints, open };
  }
  return out;
}

export async function loadSplat(base: string, m: Manifest): Promise<Object3D | null> {
  if (!m.splat) return null;
  const type = m.splatFormat === 'ply' ? SplatLoader.SplatFileType.PLY : SplatLoader.SplatFileType.SPZ;
  const data = await SplatLoader.parseSplatData(type, base + m.splat, SplatLoader.SplatPackType.Compressed);
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

async function glb(url: string, cache: Map<string, ArrayBuffer>) {
  if (!cache.has(url)) cache.set(url, await (await fetch(url)).arrayBuffer());
  return (await GLTFLoader.loadGLTF(cache.get(url)!.slice(0), { textureLoader: (u: string) => downloadTexture(u) as any })).scene;
}

/** model (glTF frame) → box frame: Y-up→Z-up, scale k, offset, then optional 90° swap about Z, then position. */
function placed(model: Object3D, k: number, pos: Vec3, swap: boolean, innerOffset: Vec3 = [0, 0, 0]) {
  model.rotation.x = Math.PI / 2;
  const inner = new Object3D(); inner.add(model); inner.scale.set(k, k, k); inner.position.set(...innerOffset);
  const outer = new Object3D(); outer.add(inner); outer.position.set(...pos);
  if (swap) outer.rotation.z = Math.PI / 2;
  return outer;
}

/** Visuals per MuJoCo body: plain Lux3D models fitted to their box; articulated models split per URDF link. */
export async function loadModels(base: string, m: Manifest, layout: Layout) {
  const cache = new Map<string, ArrayBuffer>();
  const visuals: { body: string; obj: Object3D }[] = [];
  for (const b of layout.boxes) {
    try {
      const art = m.articulated?.[b.cls];
      if (art?.joints?.length && art.links[art.root]) {
        const f = fit(assembled(art), b), c = f.c;
        visuals.push({ body: b.id, obj: placed(await glb(base + art.links[art.root].file, cache), f.k, [0, 0, 0], f.swap, [-c[0] * f.k, -c[1] * f.k, -c[2] * f.k]) });
        let i = 0;
        for (const j of art.joints) {
          const l = art.links[j.child]; if (!l) continue;
          visuals.push({ body: `${b.id}_part${i++}`, obj: placed(await glb(base + l.file, cache), f.k, toBox(j.origin, f), f.swap) });
        }
        continue;
      }
      const file = m.models?.[b.cls];
      if (!file) continue;
      const model = await glb(base + file, cache);
      model.rotation.x = Math.PI / 2;
      const inner = new Object3D(); inner.add(model);
      const bb = new Box3().setFromObject(inner);
      const size = bb.getSize(), c = bb.getCenter(new Vector3());
      const swap = (size.x > size.y) !== (b.size[0] > b.size[1]);
      const sx = swap ? size.y : size.x, sy = swap ? size.x : size.y;
      const k = Math.min(b.size[0] / (sx || 1), b.size[1] / (sy || 1), b.size[2] / (size.z || 1));
      inner.scale.set(k, k, k);
      inner.position.set(-c.x * k, -c.y * k, -c.z * k);
      const outer = new Object3D(); outer.add(inner);
      if (swap) outer.rotation.z = Math.PI / 2;
      visuals.push({ body: b.id, obj: outer });
    } catch (e) { console.warn('model', b.cls, e); }
  }
  return visuals;
}
