// Day-1 spike: prove the two unknowns on real Aholo responses.
//   A. Lux3D text→3D cabinet → articulation → what does the GLB contain? (joints? animation only?)
//   B. World generation (indoor prompt) → PLY → Asset upload → Spatial Understanding → layout text
// Resumable: state lives in spike-out/state.json; finished steps are skipped on rerun.
// Usage: pnpm spike [a|b]   (default: both)
import { createAssetClient } from '@manycore/aholo-sdk-asset';
import { createLux3dClient } from '@manycore/aholo-sdk-lux3d';
import { createWorldClient } from '@manycore/aholo-sdk-world';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const OUT = path.resolve('spike-out');
const STATE = path.join(OUT, 'state.json');
const BASE = process.env.AHOLO_API_BASE ?? 'https://api.aholo.ai/global';
const KEY = process.env.AHOLO_API_KEY;
if (!KEY) throw new Error('AHOLO_API_KEY missing — copy .env.example to .env');

await mkdir(OUT, { recursive: true });
const state = existsSync(STATE) ? JSON.parse(await readFile(STATE, 'utf8')) : {};
const save = () => writeFile(STATE, JSON.stringify(state, null, 2));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const cfg = { region: 'com', apiKey: KEY };
const lux3d = createLux3dClient(cfg);
const world = createWorldClient(cfg);
const asset = createAssetClient(cfg);
const LONG = { intervalMs: 5000, timeoutMs: 45 * 60_000 };

async function step(name, fn) {
  if (state[name] !== undefined) return state[name];
  log('→', name);
  const t0 = Date.now();
  state[name] = await fn();
  state[`${name}_ms`] = Date.now() - t0;
  await save();
  log('✓', name, `${((Date.now() - t0) / 1000).toFixed(0)}s`);
  return state[name];
}

async function download(url, file) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download ${res.status} ${url.split('?')[0]}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(path.join(OUT, file), buf);
  return buf;
}

async function rest(method, p, body) {
  const res = await fetch(`${BASE}${p}`, {
    method,
    headers: { Authorization: KEY, 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${method} ${p} → ${res.status} ${text.slice(0, 500)}`);
  return JSON.parse(text);
}

const glbOutput = (result) => {
  const o = result.outputs.find((x) => /\.glb(\?|$)/i.test(x.content ?? '')) ?? result.outputs[0];
  if (!o?.content) throw new Error(`no output URL in ${JSON.stringify(result).slice(0, 400)}`);
  return o.content;
};

// glTF 2.0 binary → JSON chunk
function glbJson(buf) {
  if (buf.readUInt32LE(0) !== 0x46546c67) throw new Error('not a GLB');
  const len = buf.readUInt32LE(12);
  return JSON.parse(buf.subarray(20, 20 + len).toString('utf8'));
}

function summarizeGltf(g) {
  return {
    asset: g.asset,
    extensionsUsed: g.extensionsUsed,
    nodeCount: g.nodes?.length,
    nodes: g.nodes?.map((n, i) => ({ i, name: n.name, mesh: n.mesh, children: n.children, t: n.translation, r: n.rotation, s: n.scale, extras: n.extras })),
    animations: g.animations?.map((a) => ({
      name: a.name,
      channels: a.channels.map((c) => ({ node: c.target.node, nodeName: g.nodes?.[c.target.node]?.name, path: c.target.path })),
    })),
    sceneExtras: g.scenes?.map((s) => s.extras),
    rootExtras: g.extras,
  };
}

async function spikeA() {
  const genTask = await step('a_text3d_task', () =>
    lux3d.textTo3d.create({
      prompt: 'A wooden kitchen wall cabinet with two hinged doors and one pull-out drawer, plain background',
      version: 'G1',
      outputFormat: ['glb'],
      aiPredictSize: true,
    }));
  const gen = await step('a_text3d_result', () => lux3d.tasks.waitFor(genTask, LONG));
  const glbUrl = glbOutput(gen);
  await step('a_text3d_glb', async () => { await download(glbUrl, 'a-cabinet.glb'); return 'a-cabinet.glb'; });

  const artTask = await step('a_artic_task', () =>
    lux3d.articulationAnimation.create({
      glbUrl,
      prompt: 'Keep the cabinet body fixed. Add a revolute hinge to each door so it swings open 90 degrees, and a prismatic joint so the drawer slides out.',
    }));
  const art = await step('a_artic_result', () => lux3d.tasks.waitFor(artTask, LONG));
  await writeFile(path.join(OUT, 'a-artic-result.json'), JSON.stringify(art, null, 2));
  for (const [i, o] of art.outputs.entries()) {
    const name = `a-artic-${i}${path.extname(new URL(o.content).pathname) || '.bin'}`;
    const buf = await download(o.content, name);
    if (name.endsWith('.glb')) {
      const summary = summarizeGltf(glbJson(buf));
      await writeFile(path.join(OUT, `${name}.summary.json`), JSON.stringify(summary, null, 2));
      log('articulated GLB:', name, 'nodes', summary.nodeCount, 'animations', summary.animations?.length ?? 0);
    } else log('articulation output:', name);
  }
}

// OpenCV (X right, Y down, Z forward) → ScanNet (Z up). Row-major, column-vector convention per spec.
const OPENCV_TO_ZUP = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]];

async function spikeB() {
  const op = await step('b_worldgen_op', () =>
    world.generations.create({
      name: 'housewarming-spike-kitchen',
      prompt: 'A small modern kitchen with a refrigerator, wall cabinets, a dining table and two chairs, a door and a window',
    }));
  const detail = await step('b_worldgen_detail', () => world.waitFor(op.worldId, LONG));
  await writeFile(path.join(OUT, 'b-world-detail.json'), JSON.stringify(detail, null, 2));
  const plyUrl = detail.assets?.splats?.urls?.plyPath;
  if (!plyUrl) throw new Error('world has no plyPath — see spike-out/b-world-detail.json');
  await step('b_ply_local', async () => {
    const buf = await download(plyUrl, 'b-world.ply');
    const header = buf.subarray(0, 4096).toString('latin1').split('end_header')[0];
    await writeFile(path.join(OUT, 'b-world.ply.header.txt'), header);
    return 'b-world.ply';
  });
  const uploaded = await step('b_ply_upload', () => asset.uploadFile(path.join(OUT, 'b-world.ply')));
  const inputFileUrl = uploaded.url;

  for (const [tag, transformationMatrix] of [['asis', undefined], ['opencv', OPENCV_TO_ZUP]]) {
    const task = await step(`b_su_${tag}_task`, () =>
      rest('POST', '/spatial-understanding/v1/tasks', { inputFileUrl, ...(transformationMatrix && { transformationMatrix }) }));
    const id = task.taskId ?? task.data?.taskId;
    const done = await step(`b_su_${tag}_result`, async () => {
      for (;;) {
        const r = await rest('GET', `/spatial-understanding/v1/tasks/${id}`);
        const t = r.data ?? r;
        if (['SUCCEEDED', 'FAILED', 'TIMEOUT'].includes(t.status)) return t;
        await new Promise((ok) => setTimeout(ok, 5000));
      }
    });
    if (done.status !== 'SUCCEEDED') { log(`spatial understanding (${tag}) ${done.status}:`, done.errorCode, done.errorMessage); continue; }
    const txt = (await download(done.result.resultSslUrl, `b-layout-${tag}.txt`)).toString('utf8');
    const counts = {};
    for (const line of txt.split('\n')) { const m = line.match(/=(\w+)\(/); if (m) counts[m[1]] = (counts[m[1]] ?? 0) + 1; }
    log(`layout (${tag}):`, JSON.stringify(counts));
  }
}

const which = process.argv[2] ?? 'ab';
if (which.includes('a')) await spikeA().catch((e) => { log('A failed:', e.message); process.exitCode = 1; });
if (which.includes('b')) await spikeB().catch((e) => { log('B failed:', e.message); process.exitCode = 1; });
