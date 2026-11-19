// Housewarming pipeline: real room → simulation bundle in public/rooms/<id>/.
//   node --env-file=.env scripts/pipeline.mjs --id kitchen --video ./kitchen.mp4
//   node --env-file=.env scripts/pipeline.mjs --id loft --image ./loft.jpg [--prompt "..."]
//   node --env-file=.env scripts/pipeline.mjs --id demo --prompt "a small kitchen ..."
// Steps (resumable, state in public/rooms/<id>/state.json):
//   1 world   Aholo World reconstruction (video) or generation (image/prompt) → 3DGS PLY/SPZ
//   2 align   up axis, floor, walls → X/Y, meters (ScanNet frame) → transform T
//   3 layout  Aholo Spatial Understanding (SpatialLM) on the aligned point cloud → walls/doors/windows/boxes
//   4 objects Lux3D text→3D per furniture class (true size comes from the box)
//   5 joints  Lux3D articulation for cabinets/wardrobes/fridges → animated GLB → hinge/slide joints
import { createAssetClient } from '@manycore/aholo-sdk-asset';
import { createLux3dClient } from '@manycore/aholo-sdk-lux3d';
import { createWorldClient } from '@manycore/aholo-sdk-world';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { readPly, splatRgb, writePointCloud } from './lib/ply.mjs';
import { alignRoom } from './lib/align.mjs';
import { jointsFromGlb } from './lib/joints.mjs';

const { values: args } = parseArgs({
  options: {
    id: { type: 'string' }, video: { type: 'string' }, insv: { type: 'string' }, image: { type: 'string' }, prompt: { type: 'string' },
    metric: { type: 'boolean', default: false }, quality: { type: 'string', default: 'normal' }, max: { type: 'string', default: '6' },
    only: { type: 'string' },
  },
});
if (!args.id) throw new Error('--id required');
const KEY = process.env.AHOLO_API_KEY; if (!KEY) throw new Error('AHOLO_API_KEY missing');
const BASE = process.env.AHOLO_API_BASE ?? 'https://api.aholo.ai/global';
const DIR = path.resolve('public/rooms', args.id);
await mkdir(path.join(DIR, 'models'), { recursive: true });
const STATE = path.join(DIR, 'state.json');
const state = existsSync(STATE) ? JSON.parse(await readFile(STATE, 'utf8')) : {};
const save = () => writeFile(STATE, JSON.stringify(state, null, 2));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const cfg = { region: 'com', apiKey: KEY };
const lux3d = createLux3dClient(cfg), world = createWorldClient(cfg), asset = createAssetClient(cfg);
const LONG = { intervalMs: 5000, timeoutMs: 60 * 60_000 };

async function step(name, fn) {
  if (state[name] !== undefined) return state[name];
  log('→', name); const t0 = Date.now();
  state[name] = await fn(); state[`${name}_s`] = Math.round((Date.now() - t0) / 1000); await save();
  log('✓', name, `${state[`${name}_s`]}s`); return state[name];
}
async function download(url, file) {
  const res = await fetch(url); if (!res.ok) throw new Error(`download ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer()); await writeFile(path.join(DIR, file), buf); return buf;
}
async function rest(method, p, body) {
  const res = await fetch(`${BASE}${p}`, { method, headers: { Authorization: KEY, 'Content-Type': 'application/json' }, body: body && JSON.stringify(body) });
  const t = await res.text(); if (!res.ok) throw new Error(`${method} ${p} ${res.status} ${t.slice(0, 300)}`); return JSON.parse(t);
}
const glbUrl = (r) => (r.outputs.find((o) => /\.glb(\?|$)/i.test(o.content ?? '')) ?? r.outputs.find((o) => o.content?.startsWith('http')))?.content;

// ---------- 1 world
async function worldStep() {
  const src = args.video ?? args.insv ?? args.image;
  const resource = src && await step('upload_src', async () => (await asset.uploadFile(path.resolve(src))).url);
  const op = await step('world_op', async () => {
    if (args.video || args.insv) {
      return world.reconstructions.create({ name: `housewarming-${args.id}`, resources: [{ url: resource, type: args.video ? 'video' : 'insv' }], taskQuality: args.quality, scene: 'space' });
    }
    return world.generations.create({ name: `housewarming-${args.id}`, ...(args.prompt && { prompt: args.prompt }), ...(resource && { resources: [{ url: resource, type: 'image' }] }) });
  });
  const detail = await step('world_detail', () => world.waitFor(op.worldId, LONG));
  const urls = detail.assets?.splats?.urls ?? {};
  await step('splat_files', async () => {
    const out = {};
    if (urls.spzPath) { await download(urls.spzPath, 'scene.spz'); out.spz = 'scene.spz'; }
    if (urls.plyPath) { await download(urls.plyPath, 'raw.ply'); out.ply = 'raw.ply'; }
    if (detail.assets?.imagery?.panoUrl) { await download(detail.assets.imagery.panoUrl, 'pano.jpg'); out.pano = 'pano.jpg'; }
    if (!out.ply) throw new Error('world has no PLY');
    return out;
  });
}

// ---------- 2 align
async function alignStep() {
  return step('align', async () => {
    const ply = readPly(path.join(DIR, 'raw.ply'));
    const x = ply.get('x'), y = ply.get('y'), z = ply.get('z');
    const op = ply.has('opacity') ? ply.get('opacity') : null;
    const rgbAll = splatRgb(ply);
    const keep = [];
    for (let i = 0; i < ply.count; i++) if (!op || op[i] > 0) keep.push(i); // sigmoid(opacity) > 0.5
    const stride = Math.max(1, Math.ceil(keep.length / 1_500_000));
    const m = Math.ceil(keep.length / stride), xyz = new Float32Array(3 * m), rgb = new Uint8Array(3 * m);
    for (let j = 0, k = 0; j < keep.length; j += stride, k++) {
      const i = keep[j]; xyz[3 * k] = x[i]; xyz[3 * k + 1] = y[i]; xyz[3 * k + 2] = z[i];
      rgb[3 * k] = rgbAll[3 * i]; rgb[3 * k + 1] = rgbAll[3 * i + 1]; rgb[3 * k + 2] = rgbAll[3 * i + 2];
    }
    const { T, points, info } = alignRoom(xyz, { metric: args.metric || !!(args.video || args.insv) && args.metric });
    writePointCloud(path.join(DIR, 'aligned.ply'), points, rgb);
    log('align', JSON.stringify(info));
    return { T, info, points: m, source: ply.count };
  });
}

// ---------- 3 layout
async function layoutStep() {
  const url = await step('aligned_upload', async () => (await asset.uploadFile(path.join(DIR, 'aligned.ply'))).url);
  const task = await step('su_task', () => rest('POST', '/spatial-understanding/v1/tasks', { inputFileUrl: url }));
  const id = task.taskId ?? task.data?.taskId;
  const done = await step('su_result', async () => {
    for (;;) {
      const r = await rest('GET', `/spatial-understanding/v1/tasks/${id}`); const t = r.data ?? r;
      if (['SUCCEEDED', 'FAILED', 'TIMEOUT'].includes(t.status)) return t;
      await new Promise((ok) => setTimeout(ok, 5000));
    }
  });
  if (done.status !== 'SUCCEEDED') throw new Error(`spatial understanding ${done.status} ${done.errorCode} ${done.errorMessage}`);
  await step('layout_file', async () => { await download(done.result.resultSslUrl, 'layout.txt'); return 'layout.txt'; });
}

// ---------- 4/5 objects + joints
const SKIP = new Set(['carpet', 'curtain', 'painting', 'chandelier', 'illumination', 'wall decoration', 'shower room', 'shower']);
const ARTICULATE = {
  wardrobe: 'Keep the wardrobe body fixed. Add a vertical revolute hinge on each door so the doors swing open outward about 100 degrees.',
  cupboard: 'Keep the cupboard body fixed. Add a vertical revolute hinge on each door so the doors swing open outward.',
  'tv cabinet': 'Keep the cabinet body fixed. Add a prismatic joint to each drawer so it slides straight out.',
  'shoe cabinet': 'Keep the cabinet body fixed. Add a hinge to each door so it swings open.',
  'bathroom cabinet': 'Keep the cabinet body fixed. Add a hinge to each door so it swings open.',
  nightstand: 'Keep the nightstand body fixed. Add a prismatic joint so the drawer slides straight out.',
  refrigerator: 'Keep the refrigerator body fixed. Add a vertical revolute hinge on the door so it swings open about 100 degrees.',
  cabinet: 'Keep the cabinet body fixed. Add a vertical revolute hinge on each door so it swings open.',
};
async function objectsStep() {
  const { parseLayout } = await import('../src/layout/spatiallm.ts');
  const layout = parseLayout(await readFile(path.join(DIR, 'layout.txt'), 'utf8'));
  const classes = [...new Set(layout.boxes.map((b) => b.cls))].filter((c) => !SKIP.has(c));
  // largest objects first: they read best on camera
  classes.sort((a, b) => vol(layout, b) - vol(layout, a));
  const pick = classes.slice(0, +args.max);
  log('classes', JSON.stringify(classes), '→ generating', JSON.stringify(pick));
  // fire all text→3D tasks first, then wait (Lux3D runs them concurrently)
  const tasks = {};
  for (const cls of pick) tasks[cls] = await step(`gen_${cls}_task`, () => lux3d.textTo3d.create({
    prompt: `A single ${cls} as found in a real home, realistic materials, whole object visible, plain background`,
    version: 'G1', outputFormat: ['glb'], aiPredictSize: true,
  }));
  const models = {};
  await Promise.all(pick.map(async (cls) => {
    try {
      const r = await step(`gen_${cls}_result`, () => lux3d.tasks.waitFor(tasks[cls], LONG));
      const url = glbUrl(r); const file = `models/${slug(cls)}.glb`;
      await step(`gen_${cls}_file`, async () => { await download(url, file); return { file, url }; });
      models[cls] = file;
    } catch (e) { log('object failed', cls, e.message); }
  }));
  const articulated = {};
  await Promise.all(pick.filter((c) => ARTICULATE[c] && state[`gen_${c}_file`]).map(async (cls) => {
    try {
      const t = await step(`art_${cls}_task`, () => lux3d.articulationAnimation.create({ glbUrl: state[`gen_${cls}_file`].url, prompt: ARTICULATE[cls] }));
      const r = await step(`art_${cls}_result`, () => lux3d.tasks.waitFor(t, LONG));
      const file = `models/${slug(cls)}-articulated.glb`;
      await step(`art_${cls}_file`, async () => { await download(glbUrl(r), file); return file; });
      articulated[cls] = { file, joints: jointsFromGlb(await readFile(path.join(DIR, file))) };
    } catch (e) { log('articulation failed', cls, e.message); }
  }));
  return { models, articulated };
}
const vol = (l, c) => Math.max(...l.boxes.filter((b) => b.cls === c).map((b) => b.size[0] * b.size[1] * b.size[2]));
const slug = (s) => s.replace(/[^a-z0-9]+/gi, '-').toLowerCase();

// ---------- run
const only = args.only?.split(',');
const run = (n) => !only || only.includes(n);
if (run('world')) await worldStep();
if (run('align')) await alignStep();
if (run('layout')) await layoutStep();
let objects = { models: {}, articulated: {} };
if (run('objects')) objects = await objectsStep();
const manifest = {
  id: args.id,
  source: args.video ? 'video' : args.insv ? 'insv' : args.image ? 'image' : 'prompt',
  splat: state.splat_files?.spz ?? state.splat_files?.ply,
  splatFormat: state.splat_files?.spz ? 'spz' : 'ply',
  pano: state.splat_files?.pano,
  T: state.align?.T,
  align: state.align?.info,
  layout: 'layout.txt',
  models: objects.models,
  articulated: objects.articulated,
  worldId: state.world_op?.worldId,
};
await writeFile(path.join(DIR, 'manifest.json'), JSON.stringify(manifest, null, 2));
log('manifest written', path.join(DIR, 'manifest.json'));
