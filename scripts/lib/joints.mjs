// Extract hinge/slide joints from an articulated (animated) GLB produced by Lux3D articulation.
// For every node animated by rotation → hinge: axis from the largest keyframe rotation, pivot = node origin.
// For every node animated by translation → slide: axis = direction of largest offset, range = its length.
// Everything is reported in the model's root frame (glTF, Y-up, model units) with the moving part's bounds.

export function parseGlb(buf) {
  if (buf.readUInt32LE(0) !== 0x46546c67) throw new Error('not a GLB');
  const jsonLen = buf.readUInt32LE(12);
  const json = JSON.parse(buf.subarray(20, 20 + jsonLen).toString('utf8'));
  const binStart = 20 + jsonLen + 8;
  const bin = buf.subarray(binStart, binStart + buf.readUInt32LE(20 + jsonLen));
  return { json, bin };
}

const COMP = { 5126: 4, 5123: 2, 5125: 4, 5121: 1 };
const NCOMP = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
function accessor(g, i) {
  const a = g.json.accessors[i], bv = g.json.bufferViews[a.bufferView];
  const n = NCOMP[a.type], start = (bv.byteOffset ?? 0) + (a.byteOffset ?? 0), stride = bv.byteStride ?? n * COMP[a.componentType];
  const out = new Float32Array(a.count * n);
  const dv = new DataView(g.bin.buffer, g.bin.byteOffset);
  for (let k = 0; k < a.count; k++) for (let c = 0; c < n; c++) {
    const o = start + k * stride + c * COMP[a.componentType];
    out[k * n + c] = a.componentType === 5126 ? dv.getFloat32(o, true) : a.componentType === 5123 ? dv.getUint16(o, true) : a.componentType === 5125 ? dv.getUint32(o, true) : dv.getUint8(o);
  }
  return { data: out, n, count: a.count, min: a.min, max: a.max };
}

// --- tiny mat4 (column-major like glTF)
const I = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function compose(t = [0, 0, 0], q = [0, 0, 0, 1], s = [1, 1, 1]) {
  const [x, y, z, w] = q, x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2, wx = w * x2, wy = w * y2, wz = w * z2;
  return [(1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0, (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0, (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0, t[0], t[1], t[2], 1];
}
function mul(a, b) { const o = new Array(16); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let v = 0; for (let k = 0; k < 4; k++) v += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = v; } return o; }
const local = (n) => (n.matrix ? [...n.matrix] : compose(n.translation, n.rotation, n.scale));
const apply = (m, p) => [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]];
const applyDir = (m, d) => { const v = [m[0] * d[0] + m[4] * d[1] + m[8] * d[2], m[1] * d[0] + m[5] * d[1] + m[9] * d[2], m[2] * d[0] + m[6] * d[1] + m[10] * d[2]]; const l = Math.hypot(...v) || 1; return v.map((x) => x / l); };

export function jointsFromGlb(buf) {
  const g = parseGlb(buf);
  const nodes = g.json.nodes ?? [];
  const parent = new Map(); nodes.forEach((n, i) => (n.children ?? []).forEach((c) => parent.set(c, i)));
  const world = (i) => { let m = local(nodes[i]), p = parent.get(i); while (p !== undefined) { m = mul(local(nodes[p]), m); p = parent.get(p); } return m; };
  const parentWorld = (i) => (parent.has(i) ? world(parent.get(i)) : I());

  // world-space bounds of all meshes (whole model) and of each node's subtree
  const subtreeBounds = (i, acc) => {
    const n = nodes[i];
    if (n.mesh !== undefined) for (const prim of g.json.meshes[n.mesh].primitives) {
      const a = g.json.accessors[prim.attributes.POSITION];
      if (!a.min) continue;
      const m = world(i);
      for (const cx of [a.min[0], a.max[0]]) for (const cy of [a.min[1], a.max[1]]) for (const cz of [a.min[2], a.max[2]]) {
        const p = apply(m, [cx, cy, cz]); for (let k = 0; k < 3; k++) { acc.min[k] = Math.min(acc.min[k], p[k]); acc.max[k] = Math.max(acc.max[k], p[k]); }
      }
    }
    for (const c of n.children ?? []) subtreeBounds(c, acc);
    return acc;
  };
  const empty = () => ({ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
  const roots = g.json.scenes?.[g.json.scene ?? 0]?.nodes ?? nodes.map((_, i) => i).filter((i) => !parent.has(i));
  const model = empty(); for (const r of roots) subtreeBounds(r, model);

  const joints = [];
  for (const anim of g.json.animations ?? []) {
    for (const ch of anim.channels) {
      const node = ch.target.node, s = anim.samplers[ch.sampler];
      const out = accessor(g, s.output);
      const pw = parentWorld(node);
      const base = nodes[node];
      const part = subtreeBounds(node, empty());
      if (ch.target.path === 'rotation') {
        // relative rotation of each key vs the rest pose, keep the largest
        const q0 = base.rotation ?? [out.data[0], out.data[1], out.data[2], out.data[3]];
        let best = { angle: 0, axis: [0, 1, 0] };
        const angles = [];
        for (let k = 0; k < out.count; k++) {
          const q = [out.data[4 * k], out.data[4 * k + 1], out.data[4 * k + 2], out.data[4 * k + 3]];
          const r = qmul(q, qinv(q0));
          const w = Math.max(-1, Math.min(1, r[3])), ang = 2 * Math.acos(Math.abs(w)), sgn = r[3] < 0 ? -1 : 1;
          const sn = Math.sqrt(1 - w * w) || 1, axis = [sgn * r[0] / sn, sgn * r[1] / sn, sgn * r[2] / sn];
          angles.push(ang);
          if (ang > best.angle) best = { angle: ang, axis };
        }
        if (best.angle < 0.05) continue;
        // axis lives in the node's parent frame (rotation composes as parent * T * R * S)
        joints.push({
          type: 'hinge', node, name: base.name ?? `node${node}`,
          pivot: apply(pw, base.translation ?? [0, 0, 0]), axis: applyDir(pw, best.axis),
          range: [0, best.angle], part,
          local: { axis: best.axis, rest: q0, translation: base.translation ?? [0, 0, 0] },
        });
      } else if (ch.target.path === 'translation') {
        const t0 = base.translation ?? [out.data[0], out.data[1], out.data[2]];
        let best = { d: 0, dir: [1, 0, 0] };
        for (let k = 0; k < out.count; k++) {
          const dv = [out.data[3 * k] - t0[0], out.data[3 * k + 1] - t0[1], out.data[3 * k + 2] - t0[2]];
          const wd = applyDir(pw, dv), wl = Math.hypot(...[pw[0] * dv[0] + pw[4] * dv[1] + pw[8] * dv[2], pw[1] * dv[0] + pw[5] * dv[1] + pw[9] * dv[2], pw[2] * dv[0] + pw[6] * dv[1] + pw[10] * dv[2]]);
          if (wl > best.d) best = { d: wl, dir: wd };
        }
        if (best.d < 1e-3) continue;
        joints.push({ type: 'slide', node, name: base.name ?? `node${node}`, pivot: apply(pw, t0), axis: best.dir, range: [0, best.d], part,
          local: { rest: t0, axis: (() => { const k = out.count - 1; const d = [out.data[3 * k] - t0[0], out.data[3 * k + 1] - t0[1], out.data[3 * k + 2] - t0[2]]; const l = Math.hypot(...d) || 1; return d.map((x) => x / l); })(), length: best.d } });
      }
    }
  }
  // de-duplicate (one joint per node, keep the larger motion)
  const byNode = new Map(); for (const j of joints) { const o = byNode.get(j.node); if (!o || j.range[1] > o.range[1]) byNode.set(j.node, j); }
  return { model, joints: [...byNode.values()], nodeCount: nodes.length, animations: (g.json.animations ?? []).length };
}

const qinv = ([x, y, z, w]) => [-x, -y, -z, w];
const qmul = ([ax, ay, az, aw], [bx, by, bz, bw]) => [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz];
