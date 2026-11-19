import { describe, expect, it } from 'vitest';
import loadMujoco from '@mujoco/mujoco';
import { strToU8, unzipSync, zipSync } from 'fflate';
import { readFile, readdir } from 'node:fs/promises';
import { parseLayout } from '../src/layout/spatiallm';
import { buildScene } from '../src/layout/mjcf';

// Builds the MJCF zip the way src/export.ts does, unpacks it into a fresh MuJoCo VFS and simulates it.
describe('MJCF export', () => {
  it('the downloaded bundle loads and steps in MuJoCo', async () => {
    const layout = parseLayout(await readFile('public/rooms/kitchen/layout.txt', 'utf8'));
    const scan = JSON.parse(await readFile('public/rooms/kitchen/colliders.json', 'utf8'));
    const xml = buildScene(layout, { robotInclude: 'lite3.xml', scan });
    const robot: [string, Uint8Array][] = [['lite3.xml', new Uint8Array(await readFile('public/robots/lite3/lite3.xml'))]];
    for (const f of await readdir('public/robots/lite3/meshes')) robot.push([`meshes/${f}`, new Uint8Array(await readFile(`public/robots/lite3/meshes/${f}`))]);
    const files: Record<string, Uint8Array> = { 'kitchen/scene.xml': strToU8(xml) };
    for (const [f, b] of robot) files[`kitchen/${f}`] = b;
    const unzipped = unzipSync(zipSync(files));

    const mj: any = await loadMujoco();
    mj.FS.mkdirTree('/x/kitchen/meshes', 0o777);
    for (const [name, bytes] of Object.entries(unzipped)) mj.FS.writeFile(`/x/${name}`, bytes);
    const m = mj.MjModel.from_xml_path('/x/kitchen/scene.xml');
    const d = new mj.MjData(m);
    try {
      expect(m.nu).toBeGreaterThanOrEqual(12);
      for (let i = 0; i < 500; i++) mj.mj_step(m, d);
      expect(Number.isFinite(d.qpos[2])).toBe(true);
    } finally { d.delete(); m.delete(); }
  }, 60_000);
});
