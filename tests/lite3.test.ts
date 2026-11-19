import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web';
import { readFile, readdir } from 'node:fs/promises';
import { parseLayout } from '../src/layout/spatiallm';
import { buildScene } from '../src/layout/mjcf';
import { Lite3Controller, gravityInBody } from '../src/robot/lite3';

// Hand-written 6 m x 4 m test room. Test fixture only.
const ROOM = `wall_0=Wall(0,0,0,6,0,0,2.6,0.0)
wall_1=Wall(6,0,0,6,4,0,2.6,0.0)
wall_2=Wall(6,4,0,0,4,0,2.6,0.0)
wall_3=Wall(0,4,0,0,0,0,2.6,0.0)
bbox_0=Bbox(dining chair,4.0,3.2,0.45,0,0.5,0.5,0.9)`;

let mj: any, model: any, data: any, ctl: Lite3Controller;

beforeAll(async () => {
  mj = await loadMujoco();
  mj.FS.mkdirTree('/work/meshes');
  const dir = 'public/robots/lite3';
  mj.FS.writeFile('/work/lite3.xml', await readFile(`${dir}/lite3.xml`, 'utf8'));
  for (const f of await readdir(`${dir}/meshes`)) mj.FS.writeFile(`/work/meshes/${f}`, await readFile(`${dir}/meshes/${f}`));
  mj.FS.writeFile('/work/scene.xml', buildScene(parseLayout(ROOM), { robotInclude: 'lite3.xml' }));
  model = mj.MjModel.from_xml_path('/work/scene.xml');
  data = new mj.MjData(model);
  const session = await ort.InferenceSession.create(await readFile('public/policy/lite3/policy.onnx'));
  ctl = new Lite3Controller(mj, model, data, session, ort);
}, 60_000);

afterAll(() => { data?.delete(); model?.delete(); });

const run = async (seconds: number) => { for (let i = 0; i < seconds / model.opt.timestep; i++) await ctl.step(); };

describe('gravityInBody', () => {
  it('is (0,0,-1) when level and flips when upside down', () => {
    expect(gravityInBody(1, 0, 0, 0)).toEqual([-0, -0, -1]);
    const [, , z] = gravityInBody(0, 1, 0, 0);
    expect(z).toBeCloseTo(1);
  });
});

describe('Lite3 in a generated room', () => {
  it('uses the 1 kHz timestep', () => expect(model.opt.timestep).toBe(0.001));

  it('stands still on a zero command', async () => {
    ctl.spawn(1.5, 2, 0);
    ctl.cmd = [0, 0, 0];
    await run(3);
    const { pos } = ctl.pose();
    expect(pos[2]).toBeGreaterThan(0.25);
    expect(Math.hypot(pos[0] - 1.5, pos[1] - 2)).toBeLessThan(0.3);
  }, 120_000);

  it('walks forward on a forward command', async () => {
    ctl.spawn(1.0, 2, 0);
    ctl.cmd = [1, 0, 0];
    await run(4);
    const { pos } = ctl.pose();
    expect(pos[2]).toBeGreaterThan(0.25);
    expect(pos[0] - 1.0).toBeGreaterThan(1.0);
  }, 120_000);

  it('is stopped by a wall', async () => {
    ctl.spawn(4.5, 1.5, 0);
    ctl.cmd = [1, 0, 0];
    await run(6);
    const { pos } = ctl.pose();
    expect(pos[0]).toBeGreaterThan(5.4); // reached the wall…
    expect(pos[0]).toBeLessThan(5.9); // …and did not pass through it
    expect(pos[2]).toBeGreaterThan(0.25); // still standing
  }, 120_000);
});
