// A room bundle produced by scripts/pipeline.mjs: splat + layout + Lux3D models + articulation joints.
import {
  Box3, GLTFLoader, Object3D, Quaternion, SplatLoader, SplatUtils, Vector3, downloadTexture,
} from '@manycore/aholo-viewer';
import type { Box, Layout, Vec3 } from './layout/spatiallm';
import type { Joint, ObjectAsset } from './layout/mjcf';
import { quatFromMat } from './render/mjview';

export interface GlbJoint {
  type: 'hinge' | 'slide'; node: number; name: string;
  pivot: number[]; axis: number[]; range: [number, number];
  part: { min: number[]; max: number[] };
  local: { axis: number[]; rest: number[]; translation?: number[]; length?: number };
}
export interface Manifest {
  id: string; source: string;
  splat?: string; splatFormat?: 'spz' | 'ply'; pano?: string;
  T?: number[][]; align?: Record<string, unknown>;
  layout: string;
  models: Record<string, string>;
  articulated: Record<string, { file: string; joints: { model: { min: number[]; max: number[] }; joints: GlbJoint[] } }>;
}

export async function loadManifest(id: string): Promise<{ manifest: Manifest; base: string; layoutText: string }> {
  const base = `/rooms/${id}/`;
  const manifest: Manifest = await (await fetch(base + 'manifest.json')).json();
  const layoutText = await (await fetch(base + manifest.layout)).text();
  return { manifest, base, layoutText };
}

/** glTF (Y-up) → MuJoCo/SpatialLM (Z-up): rotate +90° about X: (x, y, z) → (x, −z, y). */
const yUpToZUp = (p: number[]): Vec3 => [p[0], -p[2], p[1]];

/** Uniform fit of a model's (Z-up) bounds into a box: scale k and center c (model units). */
function fit(bounds: { min: number[]; max: number[] }, b: Box) {
  const a = yUpToZUp(bounds.min), z = yUpToZUp(bounds.max);
  const lo = [Math.min(a[0], z[0]), Math.min(a[1], z[1]), Math.min(a[2], z[2])];
  const hi = [Math.max(a[0], z[0]), Math.max(a[1], z[1]), Math.max(a[2], z[2])];
  const size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  // footprint may be rotated 90° relative to the box: pick the better match
  const swap = (size[0] > size[1]) !== (b.size[0] > b.size[1]);
  const sx = swap ? size[1] : size[0], sy = swap ? size[0] : size[1];
  const k = Math.min(b.size[0] / (sx || 1), b.size[1] / (sy || 1), b.size[2] / (size[2] || 1));
  return { k, c: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2] as Vec3, swap };
}
const toBox = (p: number[], f: ReturnType<typeof fit>): Vec3 => {
  const q = yUpToZUp(p); const v: Vec3 = [(q[0] - f.c[0]) * f.k, (q[1] - f.c[1]) * f.k, (q[2] - f.c[2]) * f.k];
  return f.swap ? [-v[1], v[0], v[2]] : v;
};
const dirToBox = (d: number[], f: ReturnType<typeof fit>): Vec3 => { const q = yUpToZUp(d); return f.swap ? [-q[1], q[0], q[2]] : q; };

/** MuJoCo joints (box frame, meters) for every articulated box. */
export function articulationAssets(layout: Layout, m: Manifest): Record<string, ObjectAsset> {
  const out: Record<string, ObjectAsset> = {};
  for (const b of layout.boxes) {
    const art = m.articulated[b.cls];
    if (!art?.joints?.joints?.length) continue;
    const f = fit(art.joints.model, b);
    const joints: Joint[] = art.joints.joints.map((j) => {
      const lo = toBox(j.part.min, f), hi = toBox(j.part.max, f);
      const center: Vec3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
      const size: Vec3 = [Math.max(0.02, Math.abs(hi[0] - lo[0]) * 0.96), Math.max(0.02, Math.abs(hi[1] - lo[1]) * 0.96), Math.max(0.02, Math.abs(hi[2] - lo[2]) * 0.96)];
      return {
        type: j.type, pos: toBox(j.pivot, f), axis: dirToBox(j.axis, f),
        range: j.type === 'slide' ? [0, j.range[1] * f.k] : [0, j.range[1]],
        part: { center, size },
      };
    });
    out[b.id] = { joints };
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
    // T = s·R | t  (row-major)  → position, quaternion, uniform scale
    const T = m.T, s = Math.hypot(T[0][0], T[1][0], T[2][0]);
    const R = [T[0][0] / s, T[0][1] / s, T[0][2] / s, T[1][0] / s, T[1][1] / s, T[1][2] / s, T[2][0] / s, T[2][1] / s, T[2][2] / s];
    const [x, y, z, w] = quatFromMat(R, 0);
    holder.position.set(T[0][3], T[1][3], T[2][3]);
    holder.quaternion = new Quaternion(x, y, z, w);
    holder.scale.set(s, s, s);
  }
  return holder;
}

export interface Animated { body: string; joint: string; nodes: { obj: Object3D; j: GlbJoint; k: number }[] }

/** Load each class's GLB once, fit it to every box of that class; return per-box visuals and the moving nodes. */
export async function loadModels(base: string, m: Manifest, layout: Layout) {
  const cache = new Map<string, ArrayBuffer>();
  const visuals: { box: Box; obj: Object3D; animated?: Animated }[] = [];
  for (const b of layout.boxes) {
    const art = m.articulated[b.cls];
    const file = art?.file ?? m.models[b.cls];
    if (!file) continue;
    try {
      const url = base + file;
      if (!cache.has(url)) cache.set(url, await (await fetch(url)).arrayBuffer());
      const res = await GLTFLoader.loadGLTF(cache.get(url)!.slice(0), { textureLoader: (u: string) => downloadTexture(u) as any });
      const model = res.scene;
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
      let animated: Animated | undefined;
      if (art?.joints?.joints?.length) {
        animated = { body: b.id, joint: `${b.id}_j`, nodes: [] };
        const all: Object3D[] = []; model.traverse((o) => { all.push(o); });
        for (const j of art.joints.joints) {
          const obj = all.find((o) => o.name === j.name);
          if (obj) animated.nodes.push({ obj, j, k });
        }
      }
      visuals.push({ box: b, obj: outer, animated });
    } catch (e) { console.warn('model', b.cls, e); }
  }
  return visuals;
}

/** Drive GLB door/drawer nodes from MuJoCo joint positions. */
export function poseAnimated(a: Animated, qpos: (name: string) => number) {
  a.nodes.forEach(({ obj, j, k }, i) => {
    const v = qpos(`${a.joint}${i}`);
    if (j.type === 'hinge') {
      const [ax, ay, az] = j.local.axis, h = v / 2, s = Math.sin(h);
      const r = new Quaternion(ax * s, ay * s, az * s, Math.cos(h));
      const [x, y, z, w] = j.local.rest;
      obj.quaternion = r.multiply(new Quaternion(x, y, z, w));
    } else {
      const d = v / k, [rx, ry, rz] = j.local.rest, [ax, ay, az] = j.local.axis;
      obj.position.set(rx + ax * d, ry + ay * d, rz + az * d);
    }
  });
}
