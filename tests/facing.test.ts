import { describe, expect, it } from 'vitest';
import { parseLayout } from '../src/layout/spatiallm';
import { facing } from '../src/layout/facing';

// 6 x 4 m room, Z-up. Test fixture only.
const room = (box: string) => parseLayout(`wall_0=Wall(0,0,0,6,0,0,2.6,0.0)
wall_1=Wall(6,0,0,6,4,0,2.6,0.0)
wall_2=Wall(6,4,0,0,4,0,2.6,0.0)
wall_3=Wall(0,4,0,0,0,0,2.6,0.0)
${box}`);
// front after r quarter turns, in the box frame: r0 −Y, r1 +X, r2 +Y, r3 −X
const front = (r: number, angle: number) => {
  const d = [[0, -1], [1, 0], [0, 1], [-1, 0]][r], c = Math.cos(angle), s = Math.sin(angle);
  return [c * d[0] - s * d[1], s * d[0] + c * d[1]].map((v) => Math.round(v) + 0);
};

describe('facing', () => {
  it('a fridge against the west wall faces east', () => {
    const l = room('bbox_0=Bbox(refrigerator,0.4,2,1,0,0.7,0.9,2)');
    expect(front(facing(l.boxes[0], l), 0)).toEqual([1, 0]);
  });
  it('a wardrobe against the north wall faces south, whatever its box heading', () => {
    for (const a of [0, Math.PI / 2, -Math.PI / 2, Math.PI]) {
      const l = room(`bbox_0=Bbox(wardrobe,3,3.7,1,${a},1.2,0.6,2)`);
      expect(front(facing(l.boxes[0], l), a)).toEqual([0, -1]);
    }
  });
  it('a free-standing table faces the room center', () => {
    const l = room('bbox_0=Bbox(dining table,1.6,2,0.4,0,1.2,0.8,0.75)');
    expect(front(facing(l.boxes[0], l), 0)).toEqual([1, 0]);
  });
});
