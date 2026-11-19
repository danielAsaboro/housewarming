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
import { setupRooms } from './studio';

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
  void setupRooms(id);
  const [mj, session, room, robotFiles] = await Promise.all([
    loadMujoco(),
    ort.InferenceSession.create('/policy/lite3/policy.onnx'),
    loadManifest(id),
    Promise.all(ROBOT_FILES.map(async (f) => [f, new Uint8Array(await (await fetch(`/robots/lite3/${f}`)).arrayBuffer())] as const)),
  ]);
  const { manifest, base, layoutText } = room;
  const occFile = (manifest as { occupancy?: string }).occupancy;
  if (occFile) {
    try {
      occ = await (await fetch(base + occFile)).json();
      occ!.grid = Uint8Array.from(atob(occ!.bits), (c) => c.charCodeAt(0));
    } catch { occ = null; }
  }
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
  // models stream in and attach as they arrive; the room and robot don't wait for them
  void loadModels(base, manifest, layout, (body, obj) => view.attach(body, obj));
  const A = mj.mjtObj.mjOBJ_ACTUATOR.value;

  // articulated objects: F toggles the nearest one within reach
  const doors = layout.boxes.filter((b) => objects[b.id]?.joints?.length).map((b) => ({ b, open: false }));
  const REACH = 2.2;
  const nearestDoor = () => {
    const { pos } = robot.pose();
    let best: (typeof doors)[number] | undefined, bd = REACH;
    for (const d of doors) { const dd = Math.hypot(d.b.center[0] - pos[0], d.b.center[1] - pos[1]) - Math.max(d.b.size[0], d.b.size[1]) / 2; if (dd < bd) { bd = dd; best = d; } }
    return best;
  };
  const toggleNearest = () => {
    const best = nearestDoor();
    if (!best) return;
    best.open = !best.open;
    if (best.open) view.setActive(best.b.id, true);
    else { const id = best.b.id; setTimeout(() => { if (!doors.find((d) => d.b.id === id)?.open) view.setActive(id, false); }, 1600); }
    objects[best.b.id].joints!.forEach((jt, i) => {
      const a = mj.mj_name2id(model, A, `${best!.b.id}_j${i}_act`);
      if (a >= 0) data.ctrl[a] = best!.open ? objects[best!.b.id].open[i] * 0.95 : 0;
    });
  };

  const setView = (sim: boolean) => {
    if (!splat) return;
    simView = sim; splat.visible = !sim;
    // Room: the real scan with the blueprint's linework traced over it; Blueprint: the full simulation
    view.show({ walls: true, boxes: true, models: sim }); view.linesOnly(!sim);
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
    if (k === 'r' && !keys.has('r') && !revealing) reveal();
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
  let paused = true, downSince = 0;
  const physics = async () => {
    if (paused) { simClock = performance.now(); setTimeout(physics, 16); return; }
    const ax = (k: string, k2: string) => (keys.has(k) || keys.has(k2) ? 1 : 0);
    robot.cmd = [ax('w', 'arrowup') - ax('s', 'arrowdown'), ax('q', 'q') - ax('e', 'e'), ax('a', 'arrowleft') - ax('d', 'arrowright')];
    // fall recovery: tipped over or slumped for a second → stand back up where it is
    const { pos, quat } = robot.pose();
    const up = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2]); // body z-axis · world z
    if (up < 0.6 || pos[2] < spawn[2] + 0.2) { if (!downSince) downSince = performance.now(); } else downSince = 0;
    if (downSince && performance.now() - downSince > 1000) {
      downSince = 0;
      robot.spawn(pos[0], pos[1], 2 * Math.atan2(quat[3], quat[0]), spawn[2]);
    }
    const now = performance.now();
    simClock = Math.max(simClock, now - 100); // never owe more than 100 ms
    while (simClock < now) { await robot.step(); simClock += timestep * 1000; }
    setTimeout(physics, 4);
  };
  physics();

  const p0 = robot.pose();
  let camHeading = 2 * Math.atan2(p0.quat[3], p0.quat[0]);
  const camTarget = [p0.pos[0], p0.pos[1], p0.pos[2]];
  const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
  let camDist = dist;
  // ---- build reveal: room → walls → objects → Lux3D models → robot
  const REVEAL = { walls: 1.4, boxes: 2.8, models: 4.2, drop: 5.4 };
  const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
  const smooth = (k: number) => k * k * (3 - 2 * k);
  const roomCenter = [spawn[0], spawn[1]];
  let revealing = false, revealT0 = 0, phase = -1;
  // open the reveal from the direction with the most free floor, so the camera never starts inside a wall
  const clearance = (ang: number) => {
    const dx = Math.cos(ang), dy = Math.sin(ang); let best = 8;
    for (const w of layout.walls) {
      const ex = w.b[0] - w.a[0], ey = w.b[1] - w.a[1], den = dx * ey - dy * ex;
      if (Math.abs(den) < 1e-9) continue;
      const qx = w.a[0] - spawn[0], qy = w.a[1] - spawn[1];
      const t = (qx * ey - qy * ex) / den, u = (qx * dy - qy * dx) / den;
      if (t > 0 && u >= 0 && u <= 1) best = Math.min(best, t);
    }
    return best;
  };
  let revealAngle = 0, revealDist = 3.6;
  for (let i = 0; i < 36; i++) { const ang = (i / 36) * Math.PI * 2, c = clearance(ang); if (c > clearance(revealAngle)) revealAngle = ang; }
  revealDist = Math.max(1.6, Math.min(3.6, (clearance(revealAngle) - 0.4) / Math.cos(0.42)));
  revealAngle += 0.24; // the slow orbit sweeps back across the open direction
  const step = (t: number) => {
    const s = t < REVEAL.walls ? 0 : t < REVEAL.boxes ? 1 : t < REVEAL.models ? 2 : t < REVEAL.drop ? 3 : 4;
    if (s === phase) return;
    phase = s;
    view.revealBoxes = s === 2;
    view.linesOnly(false);
    if (s === 0) { view.show({ walls: !splat, boxes: false, models: false }); view.showRobot(false); paused = true; }
    if (s === 1) view.show({ walls: true });
    if (s === 2) view.show({ boxes: true });
    if (s === 3) view.show({ boxes: false, models: true });
    if (s === 4) {
      if (splat) setView(false); else view.show({ walls: true, models: true });
      robot.spawn(spawn[0], spawn[1], 0, spawn[2] + 0.12);
      view.showRobot(true); paused = false;
    }
  };
  const reveal = () => { revealing = true; revealT0 = performance.now(); phase = -1; };
  reveal();

  const openBtn = document.getElementById('open') as HTMLButtonElement;
  let frameN = 0;
  const frame = () => {
    requestAnimationFrame(frame);
    if (++frameN % 10 === 0 && doors.length) {
      const d = nearestDoor();
      openBtn.disabled = !d;
      openBtn.firstChild!.textContent = d?.open ? 'Close' : 'Open';
    }
    view.sync();
    const { pos, quat } = robot.pose();
    // ease the camera after the robot: gait bob and turns stay smooth on screen
    const heading = 2 * Math.atan2(quat[3], quat[0]);
    camHeading += wrap(heading - camHeading) * 0.06;
    for (let i = 0; i < 3; i++) camTarget[i] += (pos[i] - camTarget[i]) * 0.12;
    camTarget[2] = camTarget[2] * 0.9 + (pos[2] + 0.05) * 0.1;
    // build reveal: high orbit over the room, then ease down behind the robot
    let d = dist, pt = pitch, a = camHeading + orbit;
    if (revealing) {
      const t = (performance.now() - revealT0) / 1000;
      step(t);
      const k = smooth(Math.min(1, Math.max(0, (t - REVEAL.drop) / 1.6)));
      // stay under the ceiling: generated and scanned rooms keep theirs
      d = lerp(revealDist, dist, k); pt = lerp(0.42, pitch, k);
      a = lerp(revealAngle - t * 0.08, camHeading + orbit, k);
      for (let i = 0; i < 2; i++) camTarget[i] = lerp(roomCenter[i], robot.pose().pos[i], k);
      if (t > REVEAL.drop + 1.7) revealing = false;
    }
    // never put the camera behind a wall or inside furniture: pull it in to the first obstacle
    const reach = rayClear(camTarget[0], camTarget[1], a, layout) - 0.25;
    camDist += (Math.max(0.6, Math.min(d, reach / Math.max(0.2, Math.cos(pt)))) - camDist) * (revealing ? 1 : 0.2);
    d = camDist;
    camera.position.set(camTarget[0] + d * Math.cos(pt) * Math.cos(a), camTarget[1] + d * Math.cos(pt) * Math.sin(a), camTarget[2] + d * Math.sin(pt));
    camera.lookAt(new Vector3(camTarget[0], camTarget[1], camTarget[2]));
    viewer.render();
  };
  requestAnimationFrame(frame);
  document.body.classList.add('ready');
  Object.assign(window, { __hw: { mj, model, data, robot, layout, viewer, view, manifest, objects, toggleNearest, sceneXml, cam: (o: number, p: number, d: number) => { orbit = o; pitch = p; dist = d; }, setView, debug: () => ({ occ: !!occ, camDist, camHeading, orbit, clear: rayClear(camTarget[0], camTarget[1], camHeading + orbit, layout) }) } });
}

function hud(m: Awaited<ReturnType<typeof loadManifest>>['manifest'] & { timings?: Record<string, number>; counts?: Record<string, number> }, l: Layout, hasSplat: boolean, hasDoors: boolean) {
  (document.getElementById('viewseg') as HTMLElement).hidden = !hasSplat;
  (document.getElementById('open') as HTMLButtonElement).disabled = true;
  (document.getElementById('open') as HTMLButtonElement).hidden = !hasDoors;
  const t = m.timings, fmt = (s?: number) => (s ? (s >= 90 ? `${Math.round(s / 60)}m` : `${s}s`) : '—');
  const stages: [string, string][] = [
    [m.source === 'video' ? 'Scan' : 'World', fmt(t?.world)],
    ['Layout', `${l.walls.length}w ${l.boxes.length}o`],
    ['Lux3D', `${Object.keys(m.models ?? {}).length}`],
    ['Joints', `${Object.values(m.articulated ?? {}).reduce((n, a) => n + (a.joints?.length ?? 0), 0)}`],
  ];
  const el = document.getElementById('build')!;
  const thumb = (m as { thumb?: string }).thumb;
  el.innerHTML = (thumb ? `<img src="/rooms/${m.id}/${thumb}" alt="">` : '') + stages.map(([a, b]) => `<div><i>${a}</i><em>${b}</em></div>`).join('');
  el.hidden = false;
}

interface Occ { origin: number[]; res: number; w: number; h: number; bits: string; grid?: Uint8Array }
let occ: Occ | null = null;
const occupied = (x: number, y: number) => {
  if (!occ) return false;
  const cx = Math.floor((x - occ.origin[0]) / occ.res), cy = Math.floor((y - occ.origin[1]) / occ.res);
  if (cx < 0 || cy < 0 || cx >= occ.w || cy >= occ.h) return false;
  const k = cy * occ.w + cx; return (occ.grid![k >> 3] >> (k & 7)) & 1;
};

/** Distance from (x, y) along heading `ang` to the first wall, tall object, or occupied scan cell (2D). */
function rayClear(x: number, y: number, ang: number, l: Layout) {
  const dx = Math.cos(ang), dy = Math.sin(ang); let best = 12;
  // scan occupancy: things the layout model missed (counters, islands); skip the robot's own cell
  if (occ) for (let t = 0.45; t < best; t += occ.res / 2) if (occupied(x + dx * t, y + dy * t)) { best = t; break; }
  const seg = (ax: number, ay: number, bx: number, by: number) => {
    const ex = bx - ax, ey = by - ay, den = dx * ey - dy * ex;
    if (Math.abs(den) < 1e-9) return;
    const qx = ax - x, qy = ay - y, t = (qx * ey - qy * ex) / den, u = (qx * dy - qy * dx) / den;
    if (t > 0 && u >= 0 && u <= 1) best = Math.min(best, t);
  };
  for (const w of l.walls) seg(w.a[0], w.a[1], w.b[0], w.b[1]);
  for (const b of l.boxes) {
    if (b.size[2] < 1.0) continue; // the camera rides above low furniture
    const c = Math.cos(b.angleZ), s = Math.sin(b.angleZ), hx = b.size[0] / 2, hy = b.size[1] / 2;
    const P = [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].map(([u, v]) => [b.center[0] + c * u - s * v, b.center[1] + s * u + c * v]);
    for (let i = 0; i < 4; i++) seg(P[i][0], P[i][1], P[(i + 1) % 4][0], P[(i + 1) % 4][1]);
  }
  return best;
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
