import { describe, expect, it } from 'vitest';
// @ts-expect-error plain JS module
import { jointsFromGlb } from '../scripts/lib/joints.mjs';

/** Build a minimal GLB: body node + door node (child) with a 1x1 quad mesh; door animated by `anim`. */
function glb(anim: { path: 'rotation' | 'translation'; keys: number[][] }, doorT = [0.4, 0, 0.3]) {
  const pos = new Float32Array([0, 0, 0, 0, 1, 0, 0, 1, 0.5]);
  const times = new Float32Array(anim.keys.map((_, i) => i));
  const vals = new Float32Array(anim.keys.flat());
  const bin = Buffer.concat([Buffer.from(pos.buffer), Buffer.from(times.buffer), Buffer.from(vals.buffer)]);
  const json = {
    asset: { version: '2.0' },
    scene: 0, scenes: [{ nodes: [0] }],
    nodes: [{ name: 'body', children: [1], mesh: 0 }, { name: 'door', mesh: 0, translation: doorT, ...(anim.path === 'rotation' ? { rotation: anim.keys[0] } : {}) }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    buffers: [{ byteLength: bin.length }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 36 }, { buffer: 0, byteOffset: 36, byteLength: times.byteLength }, { buffer: 0, byteOffset: 36 + times.byteLength, byteLength: vals.byteLength }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [0, 1, 0.5] },
      { bufferView: 1, componentType: 5126, count: times.length, type: 'SCALAR' },
      { bufferView: 2, componentType: 5126, count: anim.keys.length, type: anim.path === 'rotation' ? 'VEC4' : 'VEC3' },
    ],
    animations: [{ channels: [{ sampler: 0, target: { node: 1, path: anim.path } }], samplers: [{ input: 1, output: 2 }] }],
  };
  let j = Buffer.from(JSON.stringify(json)); j = Buffer.concat([j, Buffer.alloc((4 - (j.length % 4)) % 4, 0x20)]);
  const b = Buffer.concat([bin, Buffer.alloc((4 - (bin.length % 4)) % 4)]);
  const head = Buffer.alloc(12); head.writeUInt32LE(0x46546c67, 0); head.writeUInt32LE(2, 4); head.writeUInt32LE(12 + 8 + j.length + 8 + b.length, 8);
  const jh = Buffer.alloc(8); jh.writeUInt32LE(j.length, 0); jh.writeUInt32LE(0x4e4f534a, 4);
  const bh = Buffer.alloc(8); bh.writeUInt32LE(b.length, 0); bh.writeUInt32LE(0x004e4942, 4);
  return Buffer.concat([head, jh, j, bh, b]);
}

describe('jointsFromGlb', () => {
  it('finds a hinge: axis, pivot and swing angle', () => {
    const s = Math.SQRT1_2;
    const r = jointsFromGlb(glb({ path: 'rotation', keys: [[0, 0, 0, 1], [0, Math.sin(Math.PI / 8), 0, Math.cos(Math.PI / 8)], [0, s, 0, s]] }));
    expect(r.joints).toHaveLength(1);
    const j = r.joints[0];
    expect(j.type).toBe('hinge');
    expect(j.name).toBe('door');
    expect(j.range[1]).toBeCloseTo(Math.PI / 2, 3);
    expect(Math.abs(j.axis[1])).toBeCloseTo(1, 3);
    expect(j.pivot).toEqual([0.4, 0, 0.3]);
  });

  it('finds a slide: direction and travel', () => {
    const r = jointsFromGlb(glb({ path: 'translation', keys: [[0.4, 0, 0.3], [0.4, 0, 0.5], [0.4, 0, 0.7]] }));
    const j = r.joints[0];
    expect(j.type).toBe('slide');
    expect(j.range[1]).toBeCloseTo(0.4, 4);
    expect(j.axis[2]).toBeCloseTo(1, 4);
  });

  it('reports whole-model bounds', () => {
    const r = jointsFromGlb(glb({ path: 'rotation', keys: [[0, 0, 0, 1], [0, Math.SQRT1_2, 0, Math.SQRT1_2]] }));
    expect(r.model.max[1]).toBeCloseTo(1);
  });
});
