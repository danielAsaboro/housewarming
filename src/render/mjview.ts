// Mirrors MuJoCo geoms into an Aholo Viewer scene and keeps their transforms in sync.
import {
  BufferAttribute, BufferGeometry, LineMaterial, LineSegments, Mesh, MeshPhongMaterial, Object3D, Quaternion, Side,
} from '@manycore/aholo-viewer';

/** Blueprint palette (design.md) */
export const BLUEPRINT = { ground: 0x1b3a8f, deep: 0x10245e, line: 0xe8eefc, fill: 0x3d63c9, act: 0xd9643a };

type Mj = any;
/** geom groups drawn: 0 = room (walls, floor, objects), 2 = Lite3 visual meshes. 1 and 3 are colliders. */
const VISIBLE_GROUPS = new Set([0, 2]);

export class MjView {
  readonly root = new Object3D();
  readonly models = new Object3D();
  /** every drawn object follows a MuJoCo geom; `kind` decides its visibility */
  private items: { geom: number; obj: Object3D; kind: 'robot' | 'plane' | 'wallFill' | 'wallEdge' | 'boxFill' | 'boxEdge' | 'part'; modelled?: boolean }[] = [];
  private attached: { body: number; obj: Object3D; interactive: boolean; owner: string }[] = [];
  /** articulated objects currently opened (by box id): shown over the real scan while open */
  private active = new Set<string>();
  setActive(owner: string, on: boolean) { if (on) this.active.add(owner); else this.active.delete(owner); this.apply(); }
  private q = new Quaternion();
  private state = { walls: true, boxes: true, models: true, lines: false, robot: true };
  /** during the reveal, boxes of modelled objects show briefly before their models replace them */
  revealBoxes = false;

  constructor(private mj: Mj, private model: Mj, private data: Mj, opts: { hideRoom?: boolean } = {}) {
    this.root.add(this.models);
    const T = mj.mjtGeom;
    const m = model;
    for (let g = 0; g < m.ngeom; g++) {
      if (!VISIBLE_GROUPS.has(m.geom_group[g])) continue;
      const type = m.geom_type[g];
      const size = [m.geom_size[3 * g], m.geom_size[3 * g + 1], m.geom_size[3 * g + 2]];
      let geo: BufferGeometry | null = null;
      if (type === T.mjGEOM_BOX.value) geo = box(size[0], size[1], size[2]);
      else if (type === T.mjGEOM_PLANE.value) geo = box(size[0] || 30, size[1] || 30, 0.001);
      else if (type === T.mjGEOM_SPHERE.value) geo = sphere(size[0]);
      else if (type === T.mjGEOM_MESH.value) geo = meshGeometry(m, m.geom_dataid[g]);
      if (!geo) continue;
      const rgba = colorOf(m, g);
      if (rgba[3] < 0.05) continue;
      const room = m.geom_group[g] === 0;
      const isPlane = type === T.mjGEOM_PLANE.value;
      const isWall = room && m.geom_bodyid[g] === 0;
      const isPart = room && m.body_parentid[m.geom_bodyid[g]] !== 0;
      const color = room ? (isPlane ? BLUEPRINT.ground : isPart ? BLUEPRINT.act : BLUEPRINT.fill) : hex(rgba);
      const mat = new MeshPhongMaterial({ color, opacity: room ? (isPlane ? 1 : 0.22) : rgba[3], shininess: room ? 0 : 20 });
      mat.side = Side.DoubleSide;
      if (room && !isPlane) mat.transparent = true;
      const mesh = new Mesh(geo, mat);
      this.root.add(mesh);
      const kind = !room ? 'robot' : isPlane ? 'plane' : isPart ? 'part' : isWall ? 'wallFill' : 'boxFill';
      this.items.push({ geom: g, obj: mesh, kind });
      if (isPlane) mesh.add(grid(12, 0.5));
      if (room && !isPlane && !isPart && type === T.mjGEOM_BOX.value) {
        const edges = boxEdges(size[0], size[1], size[2]);
        this.root.add(edges);
        this.items.push({ geom: g, obj: edges, kind: isWall ? 'wallEdge' : 'boxEdge' });
      }
    }
    if (opts.hideRoom) { this.state.walls = this.state.boxes = this.state.models = false; }
    this.apply();
    this.sync();
  }

  setRoomVisible(on: boolean) { this.show({ walls: on, boxes: on, models: on }); }
  /** Room view: the blueprint's linework over the real scan (no fills, no floor). */
  linesOnly(on: boolean) { this.state.lines = on; this.apply(); }
  showRobot(on: boolean) { this.state.robot = on; this.apply(); }
  /** visibility of walls+floor, object boxes and Lux3D models (build reveal + views) */
  show(v: { walls?: boolean; boxes?: boolean; models?: boolean }) { Object.assign(this.state, Object.fromEntries(Object.entries(v).filter(([, x]) => x !== undefined))); this.apply(); }

  private apply() {
    const s = this.state;
    for (const it of this.items) {
      const modelled = it.modelled && !this.revealBoxes;
      it.obj.visible =
        it.kind === 'robot' ? s.robot :
        it.kind === 'plane' ? s.walls && !s.lines :
        it.kind === 'wallFill' ? s.walls && !s.lines :
        it.kind === 'wallEdge' ? s.walls :
        it.kind === 'boxFill' ? s.boxes && !s.lines && !modelled :
        it.kind === 'boxEdge' ? s.boxes && !modelled :
        /* part */ s.boxes && !s.lines && !modelled;
    }
    this.models.visible = true;
    for (const a of this.attached) a.obj.visible = s.models || (a.interactive && s.lines && this.active.has(a.owner));
  }

  /** Replace a body's drawn geoms with `obj` (e.g. a Lux3D model) that follows the body's pose. */
  attach(bodyName: string, obj: Object3D) {
    const m = this.model;
    const body = this.mj.mj_name2id(m, this.mj.mjtObj.mjOBJ_BODY.value, bodyName);
    if (body < 0) return false;
    for (const it of this.items) {
      const gb = m.geom_bodyid[it.geom];
      if (gb === body || m.body_parentid[gb] === body) it.modelled = true;
    }
    const holder = new Object3D();
    holder.add(obj);
    this.models.add(holder);
    // articulated objects only (a door/drawer part, or the body that owns them); free furniture stays as the real scan
    const isPart = m.body_parentid[body] !== 0 && m.body_jntnum[body] > 0 && m.jnt_type[m.body_jntadr[body]] !== 0;
    let hasMovingChild = false;
    for (let b = 0; b < m.nbody; b++) if (m.body_parentid[b] === body && m.body_jntnum[b] > 0 && m.jnt_type[m.body_jntadr[b]] !== 0) hasMovingChild = true;
    this.attached.push({ body, obj: holder, interactive: isPart || hasMovingChild, owner: bodyName.replace(/_part\d+$/, '') });
    this.apply();
    this.sync();
    return true;
  }

  sync() {
    const xp = this.data.xpos, xq = this.data.xquat;
    for (const { body: b, obj } of this.attached) {
      obj.position.set(xp[3 * b], xp[3 * b + 1], xp[3 * b + 2]);
      obj.quaternion = this.q.set(xq[4 * b + 1], xq[4 * b + 2], xq[4 * b + 3], xq[4 * b]);
    }
    const p = this.data.geom_xpos, R = this.data.geom_xmat;
    for (const { geom: g, obj } of this.items) {
      obj.position.set(p[3 * g], p[3 * g + 1], p[3 * g + 2]);
      const [x, y, z, w] = quatFromMat(R, 9 * g);
      obj.quaternion = this.q.set(x, y, z, w);
    }
  }

  dispose() {
    for (const { obj } of this.items) { this.root.remove(obj); (obj as any).freeAllGpuResourceOwned?.(); }
    this.items = [];
  }
}

function colorOf(m: Mj, g: number): number[] {
  const mat = m.geom_matid[g];
  const src = mat >= 0 && m.geom_rgba[4 * g + 3] === 1 && isDefaultGrey(m.geom_rgba, 4 * g) ? m.mat_rgba : m.geom_rgba;
  const i = src === m.mat_rgba ? 4 * mat : 4 * g;
  return [src[i], src[i + 1], src[i + 2], src[i + 3]];
}
const isDefaultGrey = (a: ArrayLike<number>, i: number) => a[i] === 0.5 && a[i + 1] === 0.5 && a[i + 2] === 0.5;

/** Row-major 3x3 rotation → quaternion (x, y, z, w). */
export function quatFromMat(m: ArrayLike<number>, o: number): [number, number, number, number] {
  const m00 = m[o], m01 = m[o + 1], m02 = m[o + 2], m10 = m[o + 3], m11 = m[o + 4], m12 = m[o + 5], m20 = m[o + 6], m21 = m[o + 7], m22 = m[o + 8];
  const tr = m00 + m11 + m22;
  if (tr > 0) { const s = 0.5 / Math.sqrt(tr + 1); return [(m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s]; }
  if (m00 > m11 && m00 > m22) { const s = 2 * Math.sqrt(1 + m00 - m11 - m22); return [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]; }
  if (m11 > m22) { const s = 2 * Math.sqrt(1 + m11 - m00 - m22); return [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]; }
  const s = 2 * Math.sqrt(1 + m22 - m00 - m11); return [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
}

function fromTriangles(pos: number[]): BufferGeometry {
  const P = Float32Array.from(pos), N = new Float32Array(P.length);
  for (let i = 0; i < P.length; i += 9) {
    const ax = P[i + 3] - P[i], ay = P[i + 4] - P[i + 1], az = P[i + 5] - P[i + 2];
    const bx = P[i + 6] - P[i], by = P[i + 7] - P[i + 1], bz = P[i + 8] - P[i + 2];
    let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
    for (let k = 0; k < 3; k++) { N[i + 3 * k] = nx; N[i + 3 * k + 1] = ny; N[i + 3 * k + 2] = nz; }
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(P, 3));
  g.setAttribute('normal', new BufferAttribute(N, 3));
  return g;
}

function box(hx: number, hy: number, hz: number) {
  const v = (x: number, y: number, z: number) => [x * hx, y * hy, z * hz];
  const quad = (a: number[], b: number[], c: number[], d: number[]) => [...a, ...b, ...c, ...a, ...c, ...d];
  return fromTriangles([
    ...quad(v(1, -1, -1), v(1, 1, -1), v(1, 1, 1), v(1, -1, 1)),
    ...quad(v(-1, 1, -1), v(-1, -1, -1), v(-1, -1, 1), v(-1, 1, 1)),
    ...quad(v(1, 1, -1), v(-1, 1, -1), v(-1, 1, 1), v(1, 1, 1)),
    ...quad(v(-1, -1, -1), v(1, -1, -1), v(1, -1, 1), v(-1, -1, 1)),
    ...quad(v(-1, -1, 1), v(1, -1, 1), v(1, 1, 1), v(-1, 1, 1)),
    ...quad(v(-1, 1, -1), v(1, 1, -1), v(1, -1, -1), v(-1, -1, -1)),
  ]);
}

function lines(pts: number[], opacity = 0.85) {
  const g = new BufferGeometry() as any;
  g.setAttribute('position', new BufferAttribute(Float32Array.from(pts), 3));
  const mat = new LineMaterial({ color: BLUEPRINT.line, opacity } as any);
  if (opacity < 1) (mat as any).transparent = true;
  return new LineSegments(g, mat);
}
function boxEdges(hx: number, hy: number, hz: number) {
  const c = (x: number, y: number, z: number) => [x * hx, y * hy, z * hz];
  const e: number[] = [];
  for (const y of [-1, 1]) for (const z of [-1, 1]) e.push(...c(-1, y, z), ...c(1, y, z));
  for (const x of [-1, 1]) for (const z of [-1, 1]) e.push(...c(x, -1, z), ...c(x, 1, z));
  for (const x of [-1, 1]) for (const y of [-1, 1]) e.push(...c(x, y, -1), ...c(x, y, 1));
  return lines(e);
}
function grid(half: number, step: number) {
  const e: number[] = [];
  for (let v = -half; v <= half + 1e-6; v += step) e.push(v, -half, 0.002, v, half, 0.002, -half, v, 0.002, half, v, 0.002);
  return lines(e, 0.18);
}

function sphere(r: number, seg = 12) {
  const out: number[] = [];
  const pt = (i: number, j: number) => {
    const th = (i / seg) * Math.PI, ph = (j / seg) * 2 * Math.PI;
    return [r * Math.sin(th) * Math.cos(ph), r * Math.sin(th) * Math.sin(ph), r * Math.cos(th)];
  };
  for (let i = 0; i < seg; i++) for (let j = 0; j < seg; j++) out.push(...pt(i, j), ...pt(i + 1, j), ...pt(i + 1, j + 1), ...pt(i, j), ...pt(i + 1, j + 1), ...pt(i, j + 1));
  return fromTriangles(out);
}

function meshGeometry(m: Mj, id: number) {
  const va = m.mesh_vertadr[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
  const V = m.mesh_vert, F = m.mesh_face;
  const out: number[] = new Array(fn * 9);
  for (let f = 0; f < fn; f++) for (let k = 0; k < 3; k++) {
    const vi = va + F[3 * (fa + f) + k];
    out[9 * f + 3 * k] = V[3 * vi]; out[9 * f + 3 * k + 1] = V[3 * vi + 1]; out[9 * f + 3 * k + 2] = V[3 * vi + 2];
  }
  return fromTriangles(out);
}

const hex = (c: number[]) => (Math.round(c[0] * 255) << 16) | (Math.round(c[1] * 255) << 8) | Math.round(c[2] * 255);
