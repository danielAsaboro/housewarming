import { describe, expect, it } from 'vitest';
// @ts-expect-error plain JS module
import { alignRoom, pct } from '../scripts/lib/align.mjs';

/** Points on the surfaces of a 5 x 4 x 2.7 m room with a sofa, in a Z-up metric frame. */
function room(): number[] {
  const p: number[] = [];
  const W = 5, D = 4, H = 2.7, s = 0.05;
  for (let x = 0; x <= W; x += s) for (let y = 0; y <= D; y += s) { p.push(x, y, 0); p.push(x, y, H); }
  for (let x = 0; x <= W; x += s) for (let z = 0; z <= H; z += s) { p.push(x, 0, z); p.push(x, D, z); }
  for (let y = 0; y <= D; y += s) for (let z = 0; z <= H; z += s) { p.push(0, y, z); p.push(W, y, z); }
  for (let x = 1; x <= 3; x += s) for (let y = 0.2; y <= 1.1; y += s) for (let z = 0; z <= 0.8; z += 0.1) p.push(x, y, z);
  return p;
}

describe('alignRoom', () => {
  it('recovers Z-up, meters and wall alignment from an OpenCV-frame, rotated, scaled room', () => {
    const src = room();
    const yaw = (20 * Math.PI) / 180, k = 0.37, c = Math.cos(yaw), s = Math.sin(yaw);
    const xyz = new Float32Array(src.length);
    for (let i = 0; i < src.length; i += 3) {
      const x = src[i], y = src[i + 1], z = src[i + 2];
      const rx = c * x - s * y, ry = s * x + c * y; // yaw
      // Z-up → OpenCV (X right, Y down, Z forward): (x, y, z) → (x, -z, y)
      xyz[i] = k * rx; xyz[i + 1] = -k * z; xyz[i + 2] = k * ry;
    }
    const { points, info } = alignRoom(xyz, { ceiling: 2.7 });
    expect(info.upAxis).toBe('y');
    const zs = new Float32Array(points.length / 3), xs = new Float32Array(points.length / 3), ys = new Float32Array(points.length / 3);
    for (let i = 0; i < zs.length; i++) { xs[i] = points[3 * i]; ys[i] = points[3 * i + 1]; zs[i] = points[3 * i + 2]; }
    expect(pct(zs, 0.02)).toBeCloseTo(0, 1);
    expect(pct(zs, 0.98)).toBeCloseTo(2.7, 1);
    // walls axis-aligned: extents along x and y equal the room's (5 and 4, in some order)
    const ex = pct(xs, 0.99) - pct(xs, 0.01), ey = pct(ys, 0.99) - pct(ys, 0.01);
    expect([ex, ey].sort().map((v) => Math.round(v * 2) / 2)).toEqual([4, 5]);
  });
});
