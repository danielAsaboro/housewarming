// Parse the URDF that Lux3D articulation returns (links with one GLB each, revolute/prismatic joints).
// Frames are glTF (Y-up, meters); every joint has rpy="0 0 0" so a child frame is the parent frame shifted to `origin`.

const attr = (s, name) => s.match(new RegExp(`${name}="([^"]*)"`))?.[1];
const nums = (s) => (s ?? '').trim().split(/\s+/).map(Number);

export function parseUrdf(xml) {
  const links = {};
  for (const m of xml.matchAll(/<link\s+name="([^"]+)"[\s\S]*?<\/link>/g)) {
    const mesh = m[0].match(/<mesh\s+filename="([^"]+)"/)?.[1];
    links[m[1]] = { mesh };
  }
  const joints = [];
  for (const m of xml.matchAll(/<joint\s+name="([^"]+)"\s+type="([^"]+)"[\s\S]*?<\/joint>/g)) {
    const body = m[0];
    const type = m[2];
    if (type !== 'revolute' && type !== 'prismatic' && type !== 'continuous') continue;
    const origin = body.match(/<origin[^>]*>/)?.[0] ?? '';
    const rpy = nums(attr(origin, 'rpy') ?? '0 0 0');
    if (rpy.some((v) => Math.abs(v) > 1e-6)) throw new Error(`joint ${m[1]}: non-zero rpy not supported`);
    const limit = body.match(/<limit[^>]*>/)?.[0] ?? '';
    joints.push({
      name: m[1], type: type === 'prismatic' ? 'slide' : 'hinge',
      parent: attr(body.match(/<parent[^>]*>/)?.[0] ?? '', 'link'),
      child: attr(body.match(/<child[^>]*>/)?.[0] ?? '', 'link'),
      origin: nums(attr(origin, 'xyz') ?? '0 0 0'),
      axis: nums(attr(body.match(/<axis[^>]*>/)?.[0] ?? '', 'xyz') ?? '1 0 0'),
      lower: +(attr(limit, 'lower') ?? (type === 'continuous' ? -Math.PI : 0)),
      upper: +(attr(limit, 'upper') ?? (type === 'continuous' ? Math.PI : 0)),
    });
  }
  const children = new Set(joints.map((j) => j.child));
  const root = Object.keys(links).find((l) => !children.has(l));
  return { root, links, joints };
}
