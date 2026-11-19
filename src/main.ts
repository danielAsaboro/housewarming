import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web/wasm';
import {
  Box3, GLTFLoader, downloadTexture, Object3D as Obj3D,
  AmbientLight, BackgroundMode, Color, DirectionalLight, PerspectiveCamera, Vector3, createViewer, setViewerConfig,
} from '@manycore/aholo-viewer';
import { parseLayout } from './layout/spatiallm';
import { buildScene } from './layout/mjcf';
import { Lite3Controller } from './robot/lite3';
import { MjView } from './render/mjview';

const ROBOT_FILES = ['lite3.xml', 'meshes/hip.STL', 'meshes/shank.STL', 'meshes/thigh.STL', 'meshes/torso.STL'];

async function layoutText(): Promise<string> {
  const url = new URLSearchParams(location.search).get('layout');
  if (url) return (await fetch(url)).text();
  return (await fetch('/rooms/sample-room.txt')).text();
}

async function main() {
  const stage = document.getElementById('stage')!;
  ort.env.wasm.wasmPaths = { wasm: '/ort/ort-wasm-simd-threaded.wasm' };
  ort.env.wasm.numThreads = 1;

  const [mj, session, layoutSrc, robotFiles] = await Promise.all([
    loadMujoco(),
    ort.InferenceSession.create('/policy/lite3/policy.onnx'),
    layoutText(),
    Promise.all(ROBOT_FILES.map(async (f) => [f, new Uint8Array(await (await fetch(`/robots/lite3/${f}`)).arrayBuffer())] as const)),
  ]);

  mj.FS.mkdirTree('/work/meshes', 0o777);
  for (const [f, bytes] of robotFiles) mj.FS.writeFile(`/work/${f}`, bytes);
  const layout = parseLayout(layoutSrc);
  mj.FS.writeFile('/work/scene.xml', buildScene(layout, { robotInclude: 'lite3.xml' }));
  const model = mj.MjModel.from_xml_path('/work/scene.xml');
  const data = new mj.MjData(model);
  const robot = new Lite3Controller(mj, model, data, session, ort);
  const spawn = spawnPoint(layout);
  robot.spawn(spawn[0], spawn[1], spawn[2]);

  const viewer = createViewer('housewarming', stage, {});
  const camera = new PerspectiveCamera(55, stage.clientWidth / stage.clientHeight, 0.05, 200);
  camera.up.set(0, 0, 1);
  viewer.setCamera(camera);
  setViewerConfig(viewer, {
    pipeline: {
      Background: { background: { active: BackgroundMode.BasicBackground, basic: { color: new Color(0.953, 0.937, 0.91) } }, ground: { enabled: false } },
      TAA: { enabled: true },
    },
  });
  const scene = viewer.getScene();
  scene.add(new AmbientLight(0xffffff, 0.55));
  const sun = new DirectionalLight(0xffffff, 0.9);
  sun.position.set(-2, -3, 6);
  scene.add(sun);
  const view = new MjView(mj, model, data);
  scene.add(view.root);
  await placeModels(view, layout);

  const keys = new Set<string>();
  stage.addEventListener('keydown', (e) => keys.add(e.key.toLowerCase()));
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
  stage.addEventListener('wheel', (e) => { dist = Math.min(8, Math.max(0.8, dist * Math.exp(e.deltaY * 0.001))); e.preventDefault(); }, { passive: false });
  addEventListener('resize', () => { viewer.resize(); camera.aspect = stage.clientWidth / stage.clientHeight; camera.updateProjectionMatrix(); });

  // Physics runs on its own timer at wall-clock rate (≈19x headroom); rendering draws whatever state is current.
  const timestep = model.opt.timestep;
  let simClock = performance.now();
  const physics = async () => {
    const ax = (k: string, k2: string) => (keys.has(k) || keys.has(k2) ? 1 : 0);
    robot.cmd = [
      ax('w', 'arrowup') - ax('s', 'arrowdown'),
      ax('q', 'q') - ax('e', 'e'),
      ax('a', 'arrowleft') - ax('d', 'arrowright'),
    ];
    if (keys.has('r')) robot.spawn(spawn[0], spawn[1], spawn[2]);
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
  Object.assign(window, { __hw: { mj, model, data, robot, layout, viewer, view } });
}

/** Lux3D models (generated in Aholo3D) drawn on the matching detected objects, scaled to each box. */
const MODELS: Record<string, string> = { 'dining chair': '/models/lux3d-dining-chair.glb', chair: '/models/lux3d-dining-chair.glb' };
async function placeModels(view: MjView, l: ReturnType<typeof parseLayout>) {
  const cache = new Map<string, ArrayBuffer>();
  for (const b of l.boxes) {
    const url = MODELS[b.cls];
    if (!url) continue;
    try {
      if (!cache.has(url)) cache.set(url, await (await fetch(url)).arrayBuffer());
      const res = await GLTFLoader.loadGLTF(cache.get(url)!.slice(0), { textureLoader: (u: string) => downloadTexture(u) as any });
      const model = res.scene;
      // glTF is Y-up; MuJoCo is Z-up
      model.rotation.x = Math.PI / 2;
      const wrap = new Obj3D(); wrap.add(model);
      wrap.updateMatrix?.();
      const box = new Box3().setFromObject(wrap);
      const size = box.getSize(); const c = box.getCenter(new (box.min.constructor as any)());
      const k = Math.min(b.size[0] / (size.x || 1), b.size[1] / (size.y || 1), b.size[2] / (size.z || 1));
      wrap.scale.set(k, k, k);
      const outer = new Obj3D(); outer.add(wrap);
      wrap.position.set(-c.x * k, -c.y * k, -c.z * k);
      view.attach(b.id, outer);
    } catch (e) { console.warn('model', b.id, e); }
  }
}

/** A free spot: room centroid, facing the longest open direction. */
function spawnPoint(l: ReturnType<typeof parseLayout>): [number, number, number] {
  if (!l.walls.length) return [0, 0, 0];
  let x = 0, y = 0;
  for (const w of l.walls) { x += w.a[0] + w.b[0]; y += w.a[1] + w.b[1]; }
  x /= 2 * l.walls.length; y /= 2 * l.walls.length;
  const blocked = (px: number, py: number) => l.boxes.some((b) => b.cls !== 'carpet' && Math.abs(px - b.center[0]) < b.size[0] / 2 + 0.4 && Math.abs(py - b.center[1]) < b.size[1] / 2 + 0.4);
  for (let r = 0; r < 3 && blocked(x, y); r += 0.25) for (let k = 0; k < 8; k++) {
    const px = x + r * Math.cos((k * Math.PI) / 4), py = y + r * Math.sin((k * Math.PI) / 4);
    if (!blocked(px, py)) { x = px; y = py; break; }
  }
  return [x, y, 0];
}

main().catch((e) => {
  console.error(e);
  document.body.dataset.error = String(e?.message ?? e);
});
