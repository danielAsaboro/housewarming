import { describe, expect, it, beforeAll } from 'vitest';
import loadMujoco from '@mujoco/mujoco';
import { parseLayout } from '../src/layout/spatiallm';
import { buildScene, wallPieces } from '../src/layout/mjcf';

// The four sample lines are copied verbatim from the Aholo spec (SpatialUnderstandingTaskResult).
const SPEC_SAMPLE = `wall_0=Wall(1.6857,-4.4625,0.0,4.6798,-4.4474,0.0,2.73,0.0)
door_0=Door(wall_0,3.1734,-4.4550,1.05,1.3852,2.1)
window_0=Window(wall_4,5.6622,-1.0338,1.65,0.6883,1.5)
bbox_0=Bbox(carpet,-3.009,4.371,-1.401,-1.5708,5.718,6.734,0.109)`;

// Hand-written test room (4 m x 3 m). Test fixture only — never shown as Aholo output.
const ROOM = `wall_0=Wall(0,0,0,4,0,0,2.6,0.0)
wall_1=Wall(4,0,0,4,3,0,2.6,0.0)
wall_2=Wall(4,3,0,0,3,0,2.6,0.0)
wall_3=Wall(0,3,0,0,0,0,2.6,0.0)
door_0=Door(wall_0,1.0,0.0,1.0,0.9,2.0)
window_0=Window(wall_1,4.0,1.5,1.5,1.2,1.0)
bbox_0=Bbox(dining chair,2.0,1.5,0.45,0.3,0.5,0.5,0.9)
bbox_1=Bbox(wardrobe,0.4,2.6,1.0,0.0,0.6,1.2,2.0)
bbox_2=Bbox(carpet,2,1.5,0.01,0,2,1.5,0.02)
something=Unknown(1,2,3)`;

describe('parseLayout', () => {
  it('parses the spec sample', () => {
    const l = parseLayout(SPEC_SAMPLE);
    expect(l.walls).toHaveLength(1);
    expect(l.walls[0]).toMatchObject({ id: 'wall_0', a: [1.6857, -4.4625, 0], height: 2.73 });
    expect(l.openings.map((o) => o.kind)).toEqual(['door', 'window']);
    expect(l.openings[1].wallId).toBe('wall_4');
    expect(l.boxes[0]).toMatchObject({ cls: 'carpet', angleZ: -1.5708, size: [5.718, 6.734, 0.109] });
  });

  it('keeps multi-word classes and skips unknown types', () => {
    const l = parseLayout(ROOM);
    expect(l.boxes[0].cls).toBe('dining chair');
    expect(l.skipped).toEqual(['something=Unknown(1,2,3)']);
  });

  it('skips malformed lines instead of failing', () => {
    const l = parseLayout('wall_0=Wall(1,2,3)\nbbox_9=Bbox(chair,x,1,1,0,1,1,1)\n');
    expect(l.walls).toHaveLength(0);
    expect(l.boxes).toHaveLength(0);
    expect(l.skipped).toHaveLength(2);
  });
});

describe('wallPieces', () => {
  const l = parseLayout(ROOM);
  it('cuts a door: left/right full-height pieces plus a lintel', () => {
    const p = wallPieces(l.walls[0], l.openings.filter((o) => o.wallId === 'wall_0'), 0.08, [2, 1.5], 0);
    // [0,0.55] full, lintel over [0.55,1.45] from z=2.0..2.6, [1.45,4] full
    expect(p).toHaveLength(3);
    const lintel = p.find((x) => x.half[2] < 0.5)!;
    expect(lintel.pos[2]).toBeCloseTo(2.3, 3);
    expect(lintel.half[0]).toBeCloseTo(0.45, 3);
  });
  it('cuts a window: sill and head pieces', () => {
    const p = wallPieces(l.walls[1], l.openings.filter((o) => o.wallId === 'wall_1'), 0.08, [2, 1.5], 0);
    expect(p).toHaveLength(4);
  });
  it('pushes slabs outward from the room', () => {
    const [p] = wallPieces(l.walls[0], [], 0.08, [2, 1.5], 0);
    expect(p.pos[1]).toBeLessThan(0); // wall_0 is y=0, room is at y>0
  });
});

describe('buildScene compiles in MuJoCo', () => {
  let mj: Awaited<ReturnType<typeof loadMujoco>>;
  beforeAll(async () => { mj = await loadMujoco(); });

  it('builds walls, a movable chair, a static wardrobe, a ghost carpet', () => {
    const xml = buildScene(parseLayout(ROOM));
    const m = mj.MjModel.from_xml_string(xml);
    const d = new mj.MjData(m);
    try {
      expect(m.njnt).toBe(1); // only the chair's free joint
      for (let i = 0; i < 1000; i++) mj.mj_step(m, d);
      // chair settles on the floor (half height 0.45) and stays inside the room
      expect(d.qpos[2]).toBeGreaterThan(0.4);
      expect(d.qpos[2]).toBeLessThan(0.5);
    } finally { d.delete(); m.delete(); }
  });

  it('adds hinge and slide joints for articulated objects', () => {
    const xml = buildScene(parseLayout(ROOM), {
      objects: {
        bbox_1: {
          joints: [
            { type: 'hinge', pos: [0.3, -0.6, 0], axis: [0, 0, 1], range: [0, 1.57], part: { center: [0.31, -0.3, 0], size: [0.02, 0.6, 1.9] } },
            { type: 'slide', pos: [0, 0, -0.8], axis: [1, 0, 0], range: [0, 0.4], part: { center: [0.2, 0, -0.8], size: [0.5, 1.0, 0.2] } },
          ],
        },
      },
    });
    const m = mj.MjModel.from_xml_string(xml);
    try {
      expect(m.njnt).toBe(3);
      expect(xml).toContain('type="hinge"');
      expect(xml).toContain('type="slide"');
    } finally { m.delete(); }
  });
});
