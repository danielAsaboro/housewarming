// Local studio: upload a room video/photo from the app, run the pipeline, stream its stages back.
//   pnpm studio   (port 5711; the Vite dev server proxies /api here)
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { createWriteStream, existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const PORT = 5711;
const ROOT = path.resolve('.');
const INPUT = path.join(ROOT, 'rooms-input');
await mkdir(INPUT, { recursive: true });
const jobs = new Map(); // id → { lines: string[], done: boolean, ok: boolean, listeners: Set<res> }

const STAGES = [
  ['upload_src', 'upload'], ['world_op', 'world'], ['world_detail', 'world'], ['splat_files', 'world'],
  ['align', 'layout'], ['su_', 'layout'], ['layout_file', 'layout'],
  ['gen_', 'objects'], ['art_', 'joints'], ['manifest written', 'done'],
];
const stageOf = (line) => STAGES.find(([k]) => line.includes(k))?.[1];

function emit(job, event) {
  const data = `data: ${JSON.stringify(event)}\n\n`;
  for (const res of job.listeners) res.write(data);
}

async function addToIndex(id) {
  const p = path.join(ROOT, 'public/rooms/index.json');
  const idx = existsSync(p) ? JSON.parse(await readFile(p, 'utf8')) : { rooms: [] };
  idx.rooms = [id, ...idx.rooms.filter((r) => r !== id)];
  await writeFile(p, JSON.stringify(idx));
}

function run(id, kind, file) {
  const job = { lines: [], done: false, ok: false, listeners: new Set(), stage: 'upload' };
  jobs.set(id, job);
  const args = ['--env-file=.env', 'scripts/pipeline.mjs', '--id', id, kind === 'video' ? '--video' : '--image', file];
  if (kind === 'image') args.push('--metric');
  const child = spawn(process.execPath, args, { cwd: ROOT });
  const onData = (buf) => {
    for (const line of buf.toString().split('\n').filter(Boolean)) {
      job.lines.push(line);
      const s = stageOf(line);
      if (s && s !== job.stage) { job.stage = s; emit(job, { stage: s }); }
      emit(job, { line });
    }
  };
  child.stdout.on('data', onData); child.stderr.on('data', onData);
  child.on('exit', async (code) => {
    job.done = true; job.ok = code === 0;
    if (job.ok) await addToIndex(id);
    emit(job, { done: true, ok: job.ok, id });
  });
}

createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  try {
    if (url.pathname === '/api/studio') { res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"studio":true}'); return; }
    if (req.method === 'POST' && url.pathname === '/api/capture') {
      // dev aid: save a browser recording (e.g. a test walk through a scan) into rooms-input/
      const name = (url.searchParams.get('name') ?? 'capture').replace(/[^a-z0-9.-]+/gi, '-');
      const file = path.join(INPUT, name);
      await new Promise((ok, fail) => req.pipe(createWriteStream(file)).on('finish', ok).on('error', fail));
      res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ file }));
      return;
    }
    if (req.method === 'POST' && url.pathname === '/api/build') {
      const name = (url.searchParams.get('name') ?? 'room').toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 32) || 'room';
      const ext = path.extname(url.searchParams.get('file') ?? '').toLowerCase();
      const kind = ['.mp4', '.mov'].includes(ext) ? 'video' : ['.jpg', '.jpeg', '.png', '.webp'].includes(ext) ? 'image' : null;
      if (!kind) { res.writeHead(400).end('video (.mp4/.mov) or photo (.jpg/.png/.webp) only'); return; }
      const id = `${name}-${Date.now().toString(36).slice(-4)}`;
      const file = path.join(INPUT, id + ext);
      await new Promise((ok, fail) => req.pipe(createWriteStream(file)).on('finish', ok).on('error', fail));
      run(id, kind, file);
      res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ id }));
      return;
    }
    const m = url.pathname.match(/^\/api\/build\/([a-z0-9-]+)\/events$/);
    if (req.method === 'GET' && m) {
      const job = jobs.get(m[1]);
      if (!job) { res.writeHead(404).end(); return; }
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
      res.write(`data: ${JSON.stringify({ stage: job.stage })}\n\n`);
      if (job.done) res.write(`data: ${JSON.stringify({ done: true, ok: job.ok, id: m[1] })}\n\n`);
      job.listeners.add(res);
      req.on('close', () => job.listeners.delete(res));
      return;
    }
    res.writeHead(404).end();
  } catch (e) {
    res.writeHead(500).end(String(e?.message ?? e));
  }
}).listen(PORT, () => console.log(`studio on http://localhost:${PORT}`));
