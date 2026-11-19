// Minimal binary_little_endian PLY reader/writer for 3DGS splats and point clouds.
import { readFileSync, writeFileSync } from 'node:fs';

const SIZES = { char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2, int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8 };
const READ = {
  char: 'getInt8', int8: 'getInt8', uchar: 'getUint8', uint8: 'getUint8', short: 'getInt16', int16: 'getInt16', ushort: 'getUint16', uint16: 'getUint16',
  int: 'getInt32', int32: 'getInt32', uint: 'getUint32', uint32: 'getUint32', float: 'getFloat32', float32: 'getFloat32', double: 'getFloat64', float64: 'getFloat64',
};

/** Returns { count, props: string[], get(name) -> Float32Array } for the first (vertex) element. */
export function readPly(path) {
  const buf = readFileSync(path);
  const end = buf.indexOf('end_header\n');
  if (end < 0) throw new Error('not a PLY');
  const header = buf.subarray(0, end).toString('latin1').split('\n');
  if (!header.some((l) => l.startsWith('format binary_little_endian'))) throw new Error('only binary_little_endian PLY supported');
  const elements = [];
  for (const l of header) {
    const p = l.trim().split(/\s+/);
    if (p[0] === 'element') elements.push({ name: p[1], count: +p[2], props: [] });
    else if (p[0] === 'property' && p[1] !== 'list') elements.at(-1).props.push({ type: p[1], name: p[2] });
    else if (p[0] === 'property' && p[1] === 'list') throw new Error('list properties not supported');
  }
  const v = elements.find((e) => e.name === 'vertex');
  let offset = end + 'end_header\n'.length;
  for (const e of elements) { if (e === v) break; offset += e.count * e.props.reduce((s, p) => s + SIZES[p.type], 0); }
  const stride = v.props.reduce((s, p) => s + SIZES[p.type], 0);
  const dv = new DataView(buf.buffer, buf.byteOffset + offset, v.count * stride);
  const offs = {}; let o = 0;
  for (const p of v.props) { offs[p.name] = { o, fn: READ[p.type] }; o += SIZES[p.type]; }
  const cache = new Map();
  return {
    count: v.count,
    props: v.props.map((p) => p.name),
    has: (n) => n in offs,
    get(name) {
      if (cache.has(name)) return cache.get(name);
      const { o: po, fn } = offs[name] ?? (() => { throw new Error(`no property ${name}`); })();
      const out = new Float32Array(v.count);
      for (let i = 0; i < v.count; i++) out[i] = dv[fn](i * stride + po, true);
      cache.set(name, out);
      return out;
    },
  };
}

/** Write xyz (+ optional uchar rgb) as a binary PLY point cloud. */
export function writePointCloud(path, xyz, rgb) {
  const n = xyz.length / 3;
  const header = ['ply', 'format binary_little_endian 1.0', `element vertex ${n}`, 'property float x', 'property float y', 'property float z',
    ...(rgb ? ['property uchar red', 'property uchar green', 'property uchar blue'] : []), 'end_header', ''].join('\n');
  const stride = 12 + (rgb ? 3 : 0);
  const body = Buffer.alloc(n * stride);
  for (let i = 0; i < n; i++) {
    body.writeFloatLE(xyz[3 * i], i * stride); body.writeFloatLE(xyz[3 * i + 1], i * stride + 4); body.writeFloatLE(xyz[3 * i + 2], i * stride + 8);
    if (rgb) { body[i * stride + 12] = rgb[3 * i]; body[i * stride + 13] = rgb[3 * i + 1]; body[i * stride + 14] = rgb[3 * i + 2]; }
  }
  writeFileSync(path, Buffer.concat([Buffer.from(header, 'latin1'), body]));
}

const SH_C0 = 0.28209479177387814;
/** 3DGS DC color → sRGB bytes. */
export function splatRgb(ply) {
  const n = ply.count, out = new Uint8Array(3 * n);
  if (ply.has('f_dc_0')) {
    const r = ply.get('f_dc_0'), g = ply.get('f_dc_1'), b = ply.get('f_dc_2');
    for (let i = 0; i < n; i++) {
      out[3 * i] = clamp255((0.5 + SH_C0 * r[i]) * 255); out[3 * i + 1] = clamp255((0.5 + SH_C0 * g[i]) * 255); out[3 * i + 2] = clamp255((0.5 + SH_C0 * b[i]) * 255);
    }
  } else if (ply.has('red')) {
    const r = ply.get('red'), g = ply.get('green'), b = ply.get('blue');
    for (let i = 0; i < n; i++) { out[3 * i] = r[i]; out[3 * i + 1] = g[i]; out[3 * i + 2] = b[i]; }
  }
  return out;
}
const clamp255 = (x) => Math.max(0, Math.min(255, Math.round(x)));
