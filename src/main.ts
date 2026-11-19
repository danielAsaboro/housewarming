import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web/wasm';
import {
  AmbientLight, BackgroundMode, Color, DirectionalLight, PerspectiveCamera, Vector3, createViewer, setViewerConfig,
} from '@manycore/aholo-viewer';
import { parseLayout, type Layout } from './layout/spatiallm';
import { buildScene } from './layout/mjcf';
import { Lite3Controller } from './robot/lite3';
import { MjView } from './render/mjview';
import { articulationAssets, loadManifest, loadModels, loadSplat } from './room';
import { exportMjcf } from './export';

const ROBOT_FILES = ['lite3.xml', 'meshes/hip.STL', 'meshes/shank.STL', 'meshes/thigh.STL', 'meshes/torso.STL'];

async function roomId() {
  const q = new URLSearchParams(location.search).get('room');
  if (q) return q;
  try { return ((await (await fetch('/rooms/index.json')).json()).rooms as string[])[0]; } catch { return 'sample'; }
}

async function main() {
  const stage = document.getElementById('stage')!;
  ort.env.wasm.wasmPaths = { wasm: '/ort/ort-wasm-simd-threaded.wasm' };
  ort.env.wasm.numThreads = 1;

  const id = await roomId();
  const [mj, session, room, robotFiles] = await Promise.all([
    loadMujoco(),
    ort.InferenceSession.create('/policy/lite3/policy.onnx'),
    loadManifest(id),
    Promise.all(ROBOT_FILES.map(async (f) => [f, new Uint8Array(await (await fetch(`/robots/lite3/${f}`)).arrayBuffer())] as const)),
  ]);
  const { manifest, base, layoutText } = room;
  const layout = parseLayout(layoutText);
  const objects = articulationAssets(layout, manifest);
  const sceneXml = buildScene(layout, { robotInclude: 'lite3.xml', objects });

  mj.FS.mkdirTree('/work/meshes', 0o777);
  for (const [f, bytes] of robotFiles) mj.FS.writeFile(`/work/${f}`, bytes);
  mj.FS.writeFile('/work/scene.xml', sceneXml);
  const model = mj.MjModel.from_xml_path('/work/scene.xml');
  const data = new mj.MjData(model);
  const robot = new Lite3Controller(mj, model, data, session, ort);
  const spawn = spawnPoint(layout);
  robot.spawn(spawn[0], spawn[1], 0, spawn[2]);

  const viewer = createViewer('housewarming', stage, {});
  const camera = new PerspectiveCamera(55, stage.clientWidth / stage.clientHeight, 0.05, 200);
  camera.up.set(0, 0, 1);
  viewer.setCamera(camera);
  setViewerConfig(viewer, {
    pipeline: {
      Background: { background: { active: BackgroundMode.BasicBackground, basic: { color: new Color(0.063, 0.141, 0.369) } }, ground: { enabled: false } },
      Splatting: { enabled: true },
      TAA: { enabled: true },
    },
  });
  const scene = viewer.getScene();
  scene.add(new AmbientLight(0xffffff, 0.55));
  const sun = new DirectionalLight(0xffffff, 0.9);
  sun.position.set(-2, -3, 6);
  scene.add(sun);

  const splat = await loadSplat(base, manifest).catch((e) => { console.warn('splat', e); return null; });
  const view = new MjView(mj, model, data, { hideRoom: !!splat });
  scene.add(view.root);
  if (splat) scene.add(splat);

  let simView = !splat;
  for (const v of await loadModels(base, manifest, layout)) view.attach(v.body, v.obj);
  const A = mj.mjtObj.mjOBJ_ACTUATOR.value;

  // articulated objects: F toggles the nearest one within reach
  const doors = layout.boxes.filter((b) => objects[b.id]?.joints?.length).map((b) => ({ b, open: false }));
  const toggleNearest = () => {
    const { pos } = robot.pose();
    let best: (typeof doors)[number] | undefined, bd = 2.0;
    for (const d of doors) { const dd = Math.hypot(d.b.center[0] - pos[0], d.b.center[1] - pos[1]); if (dd < bd) { bd = dd; best = d; } }
    if (!best) return;
    best.open = !best.open;
    objects[best.b.id].joints!.forEach((jt, i) => {
      const a = mj.mj_name2id(model, A, `${best!.b.id}_j${i}_act`);
      if (a >= 0) data.ctrl[a] = best!.open ? objects[best!.b.id].open[i] * 0.95 : 0;
    });
  };

  const setView = (sim: boolean) => {
    if (!splat) return;
    simView = sim; splat.visible = !sim; view.setRoomVisible(sim);
    document.querySelectorAll<HTMLButtonElement>('#viewseg button').forEach((b) => b.setAttribute('aria-pressed', String((b.dataset.view === 'blueprint') === sim)));
  };
  hud(manifest, layout, !!splat, doors.length > 0);
  document.querySelectorAll<HTMLButtonElement>('#viewseg button').forEach((b) => b.addEventListener('click', () => { setView(b.dataset.view === 'blueprint'); stage.focus(); }));
  document.getElementById('open')!.addEventListener('click', () => { toggleNearest(); stage.focus(); });
  document.getElementById('export')!.addEventListener('click', () => { exportMjcf(id, sceneXml, robotFiles); stage.focus(); });

  const keys = new Set<string>();
  stage.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === 'f' && !keys.has('f')) toggleNearest();
    if (k === 'm' && !keys.has('m')) exportMjcf(id, sceneXml, robotFiles);
    if (k === 'v' && !keys.has('v')) setView(!simView);
    keys.add(k);
  });
  stage.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
  stage.focus();

  // follow camera; drag to orbit, wheel to zoom
  let orbit = Math.PI, pitch = 0.45, dist = 2.4, dragging = false, lastX = 0, lastY = 0;
  stage.addEventListener('pointerdown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; stage.setPointerCapture(e.pointerId); stage.focus(); });
  stage.addEventListener('pointerup', () => { dragging = false; });
  stage.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    orbit -= (e.clientX - lastX) * 0.006; pitch = Math.min(1.3, Math.max(0.05, pitch + (e.clientY - lastY) * 0.005));
    lastX = e.clientX; lastY = e.clientY;
  });
  stage.addEventListener('wheel', (e) => { dist = Math.min(10, Math.max(0.8, dist * Math.exp(e.deltaY * 0.001))); e.preventDefault(); }, { passive: false });
  addEventListener('resize', () => { viewer.resize(); camera.aspect = stage.clientWidth / stage.clientHeight; camera.updateProjectionMatrix(); });

  // Physics runs on its own timer at wall-clock rate (≈19x headroom); rendering draws whatever state is current.
  const timestep = model.opt.timestep;
  let simClock = performance.now();
  const physics = async () => {
    const ax = (k: string, k2: string) => (keys.has(k) || keys.has(k2) ? 1 : 0);
    robot.cmd = [ax('w', 'arrowup') - ax('s', 'arrowdown'), ax('q', 'q') - ax('e', 'e'), ax('a', 'arrowleft') - ax('d', 'arrowright')];
    if (keys.has('r')) robot.spawn(spawn[0], spawn[1], 0, spawn[2]);
    const now = performance.now();
    simClock = Math.max(simClock, now - 100); // never owe more than 100 ms
    while (simClock < now) { await robot.step(); simClock += timestep * 1000; }
    setTimeout(physics, 4);
  };
  physics();

  const frame = () => {
    requestAnimationFrame(frame);
    view.sync();
    const { pos, quat } = robot.pose();
    const heading = 2 * Math.atan2(quat[3], quat[0]);
    const a = heading + orbit;
    camera.position.set(pos[0] + dist * Math.cos(pitch) * Math.cos(a), pos[1] + dist * Math.cos(pitch) * Math.sin(a), pos[2] + dist * Math.sin(pitch));
    camera.lookAt(new Vector3(pos[0], pos[1], pos[2]));
    viewer.render();
  };
  requestAnimationFrame(frame);
  document.body.classList.add('ready');
  Object.assign(window, { __hw: { mj, model, data, robot, layout, viewer, view, manifest, objects, toggleNearest, sceneXml } });
}

function hud(m: Awaited<ReturnType<typeof loadManifest>>['manifest'] & { timings?: Record<string, number>; counts?: Record<string, number> }, l: Layout, hasSplat: boolean, hasDoors: boolean) {
  document.getElementById('room-name')!.textContent = m.id;
  (document.getElementById('viewseg') as HTMLElement).hidden = !hasSplat;
  (document.getElementById('open') as HTMLButtonElement).disabled = !hasDoors;
  const t = m.timings, fmt = (s?: number) => (s ? (s >= 90 ? `${Math.round(s / 60)}m` : `${s}s`) : '—');
  const stages: [string, string][] = [
    [m.source === 'video' ? 'Scan' : 'World', fmt(t?.world)],
    ['Layout', `${l.walls.length}w ${l.boxes.length}o`],
    ['Lux3D', `${Object.keys(m.models ?? {}).length}`],
    ['Joints', `${Object.values(m.articulated ?? {}).reduce((n, a) => n + (a.joints?.length ?? 0), 0)}`],
  ];
  const el = document.getElementById('build')!;
  el.innerHTML = stages.map(([a, b]) => `<div><i>${a}</i><em>${b}</em></div>`).join('');
  el.hidden = false;
}

/** A free spot near the room's center. */
function spawnPoint(l: Layout): [number, number, number] {
  if (!l.walls.length) return [0, 0, 0];
  let x = 0, y = 0;
  for (const w of l.walls) { x += w.a[0] + w.b[0]; y += w.a[1] + w.b[1]; }
  x /= 2 * l.walls.length; y /= 2 * l.walls.length;
  const blocked = (px: number, py: number) => l.boxes.some((b) => b.cls !== 'carpet' && Math.abs(px - b.center[0]) < b.size[0] / 2 + 0.4 && Math.abs(py - b.center[1]) < b.size[1] / 2 + 0.4);
  for (let r = 0; r < 3 && blocked(x, y); r += 0.25) for (let k = 0; k < 8; k++) {
    const px = x + r * Math.cos((k * Math.PI) / 4), py = y + r * Math.sin((k * Math.PI) / 4);
    if (!blocked(px, py)) { x = px; y = py; break; }
  }
  const floor = [...l.walls.map((w) => Math.min(w.a[2], w.b[2]))].sort((a, b) => a - b)[Math.floor(l.walls.length / 2)];
  return [x, y, floor];
}

main().catch((e) => {
  console.error(e);
  document.body.dataset.error = String(e?.message ?? e);
});
