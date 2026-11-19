import type { Box, Layout } from './spatiallm';

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

