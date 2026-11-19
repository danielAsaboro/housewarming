// Align an indoor point cloud to the ScanNet convention Spatial Understanding expects:
// meters, right-handed, Z up, floor at z=0, main walls parallel to X/Y.
// Returns the 4x4 row-major transform T (p' = T p) and the transformed points.

/** @param {Float32Array} xyz flat xyz  @param {{metric?: boolean, ceiling?: number}} opts */
export function alignRoom(xyz, opts = {}) {
  const n = xyz.length / 3;
  // 1. up axis: the axis whose extremes are densest (floor + ceiling planes)
  const axes = [0, 1, 2].map((a) => {
    const v = new Float32Array(n);
    for (let i = 0; i < n; i++) v[i] = xyz[3 * i + a];
    const lo = pct(v, 0.02), hi = pct(v, 0.98), span = hi - lo || 1;
    let nearLo = 0, nearHi = 0, below = 0;
    for (let i = 0; i < n; i++) {
      const t = (v[i] - lo) / span;
      if (t < 0.04) nearLo++; else if (t > 0.96) nearHi++;
      if (t < 0.5) below++;
    }
    return { a, lo, hi, plane: (nearLo + nearHi) / n, below: below / n, nearLo, nearHi };
  });
  const up = axes.reduce((b, x) => (x.plane > b.plane ? x : b));
  // sign: furniture sits on the floor, so the floor half holds more points
  const sign = up.below >= 0.5 ? 1 : -1; // +1: floor at low values

  // rotation R0 mapping (up axis, sign) → +Z, keeping a right-handed frame
  const R0 = upToZ(up.a, sign);
  const p = new Float32Array(xyz.length);
  for (let i = 0; i < n; i++) {
    const x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    p[3 * i] = R0[0] * x + R0[1] * y + R0[2] * z;
    p[3 * i + 1] = R0[3] * x + R0[4] * y + R0[5] * z;
    p[3 * i + 2] = R0[6] * x + R0[7] * y + R0[8] * z;
  }

  // 2. yaw: rotate so wall points line up with X/Y (maximize histogram peakiness of a mid-height band)
  const zs = new Float32Array(n); for (let i = 0; i < n; i++) zs[i] = p[3 * i + 2];
  const z0 = pct(zs, 0.02), z1 = pct(zs, 0.98);
  const band = [];
  for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 200000))) {
    const t = (p[3 * i + 2] - z0) / (z1 - z0 || 1);
    if (t > 0.35 && t < 0.85) band.push(p[3 * i], p[3 * i + 1]);
  }
  const scores = [];
  for (let deg = 0; deg < 90; deg += 1) {
    const th = (deg * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th);
    const hx = new Map(), hy = new Map(), bin = 0.05;
    for (let i = 0; i < band.length; i += 2) {
      const x = c * band[i] - s * band[i + 1], y = s * band[i] + c * band[i + 1];
      const kx = Math.round(x / bin), ky = Math.round(y / bin);
      hx.set(kx, (hx.get(kx) ?? 0) + 1); hy.set(ky, (hy.get(ky) ?? 0) + 1);
    }
    let score = 0; for (const v of hx.values()) score += v * v; for (const v of hy.values()) score += v * v;
    scores.push({ deg, th, score });
  }
  // prefer the angle closest to the source axes among near-ties (sources are usually already square)
  const top = Math.max(...scores.map((x) => x.score));
  const near = scores.filter((x) => x.score >= 0.98 * top);
  const best = near.reduce((b, x) => (Math.min(x.deg, 90 - x.deg) < Math.min(b.deg, 90 - b.deg) ? x : b));
  if (best.deg > 45 && 90 - best.deg <= 1) { best.deg = 0; best.th = 0; }
  const c = Math.cos(best.th), s = Math.sin(best.th);
  const Rz = [c, -s, 0, s, c, 0, 0, 0, 1];

  // 3. scale: metric sources keep scale; otherwise make floor→ceiling the expected room height
  const height = z1 - z0;
  const scale = opts.metric ? 1 : (opts.ceiling ?? 2.7) / (height || 1);

  // compose: T = S · Rz · R0, then translate floor to 0 and center XY
  const R = mul3(Rz, R0);
  const out = new Float32Array(xyz.length);
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) {
    const x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    out[3 * i] = scale * (R[0] * x + R[1] * y + R[2] * z);
    out[3 * i + 1] = scale * (R[3] * x + R[4] * y + R[5] * z);
    out[3 * i + 2] = scale * (R[6] * x + R[7] * y + R[8] * z);
  }
  const xs = new Float32Array(n), ys = new Float32Array(n), zz = new Float32Array(n);
  for (let i = 0; i < n; i++) { xs[i] = out[3 * i]; ys[i] = out[3 * i + 1]; zz[i] = out[3 * i + 2]; }
  mx = (pct(xs, 0.02) + pct(xs, 0.98)) / 2; my = (pct(ys, 0.02) + pct(ys, 0.98)) / 2;
  const floor = pct(zz, 0.02);
  for (let i = 0; i < n; i++) { out[3 * i] -= mx; out[3 * i + 1] -= my; out[3 * i + 2] -= floor; }

  const T = [
    [scale * R[0], scale * R[1], scale * R[2], -mx],
    [scale * R[3], scale * R[4], scale * R[5], -my],
    [scale * R[6], scale * R[7], scale * R[8], -floor],
    [0, 0, 0, 1],
  ];
  return { T, points: out, info: { upAxis: 'xyz'[up.a], upSign: sign, yawDeg: (best.th * 180) / Math.PI, scale, rawHeight: height, metric: !!opts.metric } };
}

function upToZ(a, sign) {
  // rows of R: new x, new y, new z (= sign * axis a)
  if (a === 2) return sign > 0 ? [1, 0, 0, 0, 1, 0, 0, 0, 1] : [1, 0, 0, 0, -1, 0, 0, 0, -1];
  if (a === 1) return sign > 0 ? [1, 0, 0, 0, 0, -1, 0, 1, 0] : [1, 0, 0, 0, 0, 1, 0, -1, 0];
  return sign > 0 ? [0, 1, 0, 0, 0, 1, 1, 0, 0] : [0, -1, 0, 0, 0, 1, -1, 0, 0];
}
function mul3(A, B) {
  const C = new Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) C[3 * r + c] = A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c];
  return C;
}
export function pct(arr, q) {
  const step = Math.max(1, Math.floor(arr.length / 100000));
  const s = []; for (let i = 0; i < arr.length; i += step) s.push(arr[i]);
  s.sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}
