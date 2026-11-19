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

/**
 * Static colliders for furniture the layout model missed (islands, counters): dense cells at furniture
 * height, away from walls and outside detected boxes, merged into rectangles with a height from the points.
 * Floaters are suppressed by a density threshold and a minimum footprint.
 */
export function sceneColliders(points, layout, { res = 0.1, z0 = 0.25, z1 = 1.6, minPts = 6, wallGap = 0.2, minArea = 0.12 } = {}) {
  const n = points.length / 3;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) { const x = points[3 * i], y = points[3 * i + 1]; if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y; }
  const w = Math.ceil((x1 - x0) / res) + 1, h = Math.ceil((y1 - y0) / res) + 1;
  const count = new Uint16Array(w * h), top = new Float32Array(w * h);
  for (let i = 0; i < n; i++) {
    const z = points[3 * i + 2]; if (z < z0 || z > z1) continue;
    const k = Math.floor((points[3 * i + 1] - y0) / res) * w + Math.floor((points[3 * i] - x0) / res);
    if (count[k] < 65535) count[k]++; if (z > top[k]) top[k] = z;
  }
  const cellCenter = (cx, cy) => [x0 + (cx + 0.5) * res, y0 + (cy + 0.5) * res];
  const nearWall = (x, y) => layout.walls.some((wl) => {
    const ex = wl.b[0] - wl.a[0], ey = wl.b[1] - wl.a[1], L2 = ex * ex + ey * ey || 1;
    const t = Math.max(0, Math.min(1, ((x - wl.a[0]) * ex + (y - wl.a[1]) * ey) / L2));
    return Math.hypot(x - (wl.a[0] + t * ex), y - (wl.a[1] + t * ey)) < wallGap;
  });
  const inBox = (x, y) => layout.boxes.some((b) => {
    const c = Math.cos(b.angleZ), s = Math.sin(b.angleZ), dx = x - b.center[0], dy = y - b.center[1];
    return Math.abs(c * dx + s * dy) < b.size[0] / 2 + 0.1 && Math.abs(-s * dx + c * dy) < b.size[1] / 2 + 0.1;
  });
  const free = new Uint8Array(w * h);
  for (let cy = 0; cy < h; cy++) for (let cx = 0; cx < w; cx++) {
    const k = cy * w + cx; if (count[k] < minPts) continue;
    const [x, y] = cellCenter(cx, cy); if (nearWall(x, y) || inBox(x, y)) continue;
    free[k] = 1;
  }
  // greedy rectangle merge
  const out = [];
  for (let cy = 0; cy < h; cy++) for (let cx = 0; cx < w; cx++) {
    if (!free[cy * w + cx]) continue;
    let ex = cx; while (ex + 1 < w && free[cy * w + ex + 1]) ex++;
    let ey = cy; grow: while (ey + 1 < h) { for (let x = cx; x <= ex; x++) if (!free[(ey + 1) * w + x]) break grow; ey++; }
    const tops = [];
    for (let y = cy; y <= ey; y++) for (let x = cx; x <= ex; x++) { free[y * w + x] = 0; tops.push(top[y * w + x]); }
    const sx = (ex - cx + 1) * res, sy = (ey - cy + 1) * res;
    if (sx * sy < minArea) continue;
    tops.sort((a, b) => a - b);
    const hgt = tops[Math.floor(tops.length * 0.7)];
    out.push({ center: [x0 + (cx + ex + 1) / 2 * res, y0 + (cy + ey + 1) / 2 * res], size: [sx, sy], height: +hgt.toFixed(3) });
  }
  return out;
}
