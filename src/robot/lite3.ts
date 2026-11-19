// Port of DeepRoboticsLab/Lite3_rl_deploy run_policy/lite3_test_policy_runner_onnx.hpp (BSD-3)
// and interface/robot/simulation/mujoco_simulation.py PD law, for MuJoCo WASM + onnxruntime-web.
//   PD at 1 kHz (kp 30, kd 1), policy every 12 physics steps.
//   obs(45) = [ω_body·0.25, gravity_body, cmd·0.8, q−q0, q̇·0.05, last_action]
//   target = q0 + action ⊙ scale
import type { InferenceSession, Tensor } from 'onnxruntime-web';

type Mj = any; // @mujoco/mujoco module; its typings are large, keep this file decoupled
type Ort = { Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => Tensor };

export const JOINTS = [
  'FL_HipX_joint', 'FL_HipY_joint', 'FL_Knee_joint',
  'FR_HipX_joint', 'FR_HipY_joint', 'FR_Knee_joint',
  'HL_HipX_joint', 'HL_HipY_joint', 'HL_Knee_joint',
  'HR_HipX_joint', 'HR_HipY_joint', 'HR_Knee_joint',
] as const;
const Q0 = Float32Array.from([0, -0.65, 1.3, 0, -0.65, 1.3, 0, -0.65, 1.3, 0, -0.65, 1.3]);
const ACTION_SCALE = Float32Array.from([0.125, 0.25, 0.25, 0.125, 0.25, 0.25, 0.125, 0.25, 0.25, 0.125, 0.25, 0.25]);
const KP = 30, KD = 1, TAU_MAX = 30, DECIMATION = 12, MAX_CMD = 0.8;
export const LITE3_TIMESTEP = 0.001;
export const STAND_HEIGHT = 0.36;

export class Lite3Controller {
  readonly base: { qpos: number; qvel: number };
  private qadr: number[];
  private dadr: number[];
  private act: number[];
  private target = Float32Array.from(Q0);
  private lastAction = new Float32Array(12);
  private obs = new Float32Array(45);
  private steps = 0;
  /** normalized command in [-1,1]: forward, left, yaw-left */
  cmd: [number, number, number] = [0, 0, 0];
  /** seconds of PD-only hold before the policy takes over */
  holdSeconds = 0.4;

  constructor(private mj: Mj, private model: Mj, private data: Mj, private session: InferenceSession, private ort: Ort) {
    const id = (type: number, name: string) => {
      const i = mj.mj_name2id(model, type, name);
      if (i < 0) throw new Error(`Lite3: missing ${name}`);
      return i;
    };
    const J = mj.mjtObj.mjOBJ_JOINT.value, A = mj.mjtObj.mjOBJ_ACTUATOR.value;
    const fb = id(J, 'floating_base');
    this.base = { qpos: model.jnt_qposadr[fb], qvel: model.jnt_dofadr[fb] };
    this.qadr = JOINTS.map((n) => model.jnt_qposadr[id(J, n)]);
    this.dadr = JOINTS.map((n) => model.jnt_dofadr[id(J, n)]);
    this.act = JOINTS.map((n) => id(A, `${n}_ctrl`));
  }

  /** Place the robot standing at (x, y) facing yaw, joints at the policy's default pose. */
  spawn(x: number, y: number, yaw = 0, floorZ = 0) {
    const q = this.data.qpos, v = this.data.qvel, b = this.base.qpos;
    q[b] = x; q[b + 1] = y; q[b + 2] = floorZ + STAND_HEIGHT;
    q[b + 3] = Math.cos(yaw / 2); q[b + 4] = 0; q[b + 5] = 0; q[b + 6] = Math.sin(yaw / 2);
    for (let i = 0; i < 6; i++) v[this.base.qvel + i] = 0;
    this.qadr.forEach((a, i) => (q[a] = Q0[i]));
    this.dadr.forEach((a) => (v[a] = 0));
    this.target.set(Q0); this.lastAction.fill(0); this.steps = 0;
    this.mj.mj_forward(this.model, this.data);
  }

  /** Advance one physics step; runs the policy every DECIMATION steps. */
  async step() {
    if (this.steps % DECIMATION === 0 && this.steps * this.model.opt.timestep >= this.holdSeconds) await this.policy();
    const q = this.data.qpos, v = this.data.qvel, ctrl = this.data.ctrl;
    for (let i = 0; i < 12; i++) {
      const tau = KP * (this.target[i] - q[this.qadr[i]]) - KD * v[this.dadr[i]];
      ctrl[this.act[i]] = Math.max(-TAU_MAX, Math.min(TAU_MAX, tau));
    }
    this.mj.mj_step(this.model, this.data);
    this.steps++;
  }

  private async policy() {
    const q = this.data.qpos, v = this.data.qvel, o = this.obs;
    const b = this.base.qpos, d = this.base.qvel;
    // free-joint angular velocity (qvel[3:6]) is already in the body frame
    o[0] = v[d + 3] * 0.25; o[1] = v[d + 4] * 0.25; o[2] = v[d + 5] * 0.25;
    const [gx, gy, gz] = gravityInBody(q[b + 3], q[b + 4], q[b + 5], q[b + 6]);
    o[3] = gx; o[4] = gy; o[5] = gz;
    o[6] = this.cmd[0] * MAX_CMD; o[7] = this.cmd[1] * MAX_CMD; o[8] = this.cmd[2] * MAX_CMD;
    for (let i = 0; i < 12; i++) {
      o[9 + i] = q[this.qadr[i]] - Q0[i];
      o[21 + i] = v[this.dadr[i]] * 0.05;
      o[33 + i] = this.lastAction[i];
    }
    const out = await this.session.run({ obs: new this.ort.Tensor('float32', o, [1, 45]) });
    const a = out.actions.data as Float32Array;
    for (let i = 0; i < 12; i++) {
      this.lastAction[i] = a[i];
      this.target[i] = Q0[i] + a[i] * ACTION_SCALE[i];
    }
  }

  pose() {
    const q = this.data.qpos, b = this.base.qpos;
    return { pos: [q[b], q[b + 1], q[b + 2]] as const, quat: [q[b + 3], q[b + 4], q[b + 5], q[b + 6]] as const };
  }
}

/** R(q)ᵀ · (0,0,−1) for a unit quaternion (w,x,y,z). */
export function gravityInBody(w: number, x: number, y: number, z: number): [number, number, number] {
  // third row of R, negated: R[2][*] = [2(xz−wy), 2(yz+wx), 1−2(x²+y²)]
  return [-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))];
}
