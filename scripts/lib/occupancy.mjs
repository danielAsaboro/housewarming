// 2D occupancy from the aligned room point cloud (Z-up meters, floor at 0): which 10 cm cells hold
// something between knee and head height. Used by the follow camera to stay out of furniture the
// layout model missed.
export function occupancy(points, { res = 0.1, z0 = 0.15, z1 = 1.9, minPts = 6 } = {}) {
  const n = points.length / 3;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = points[3 * i], y = points[3 * i + 1];
    if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y;
  }
  const w = Math.ceil((x1 - x0) / res) + 1, h = Math.ceil((y1 - y0) / res) + 1;
  const count = new Uint16Array(w * h);
  for (let i = 0; i < n; i++) {
    const z = points[3 * i + 2];
    if (z < z0 || z > z1) continue;
    const cx = Math.floor((points[3 * i] - x0) / res), cy = Math.floor((points[3 * i + 1] - y0) / res);
    const k = cy * w + cx; if (count[k] < 65535) count[k]++;
  }
  const bits = new Uint8Array(Math.ceil((w * h) / 8));
  let occupied = 0;
  for (let k = 0; k < w * h; k++) if (count[k] >= minPts) { bits[k >> 3] |= 1 << (k & 7); occupied++; }
  return { origin: [x0, y0], res, w, h, bits: Buffer.from(bits).toString('base64'), occupied };
}
