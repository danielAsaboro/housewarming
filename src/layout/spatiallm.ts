// Parser for Aholo Spatial Understanding (SpatialLM) result text.
// Spec: openapi.en.json → SpatialUnderstandingTaskResult.resultSslUrl.
// Meters, right-handed, Z-up. One entity per line: `{id}={Type}(args)`. Unknown types are skipped.

export interface Wall { id: string; a: Vec3; b: Vec3; height: number; thickness: number }
export interface Opening { id: string; kind: 'door' | 'window'; wallId: string; center: Vec3; width: number; height: number }
export interface Box { id: string; cls: string; center: Vec3; angleZ: number; size: Vec3 }
export type Vec3 = [number, number, number];

export interface Layout { walls: Wall[]; openings: Opening[]; boxes: Box[]; skipped: string[] }

const LINE = /^\s*([A-Za-z0-9_]+)\s*=\s*([A-Za-z]+)\((.*)\)\s*$/;

export function parseLayout(text: string): Layout {
  const out: Layout = { walls: [], openings: [], boxes: [], skipped: [] };
  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim()) continue;
    const m = raw.match(LINE);
    if (!m) { out.skipped.push(raw); continue; }
    const [, id, type, body] = m;
    const parts = body.split(',').map((s) => s.trim());
    try {
      switch (type) {
        case 'Wall': {
          const n = nums(parts, 8);
          out.walls.push({ id, a: [n[0], n[1], n[2]], b: [n[3], n[4], n[5]], height: n[6], thickness: n[7] });
          break;
        }
        case 'Door':
        case 'Window': {
          const [wallId, ...rest] = parts;
          const n = nums(rest, 5);
          out.openings.push({ id, kind: type === 'Door' ? 'door' : 'window', wallId, center: [n[0], n[1], n[2]], width: n[3], height: n[4] });
          break;
        }
        case 'Bbox': {
          // class may contain spaces or commas: take the last 7 numbers, the rest is the class.
          if (parts.length < 8) throw new Error('short Bbox');
          const n = nums(parts.slice(-7), 7);
          const cls = parts.slice(0, -7).join(',').replace(/^["']|["']$/g, '').replace(/_/g, ' ').trim();
          out.boxes.push({ id, cls, center: [n[0], n[1], n[2]], angleZ: n[3], size: [n[4], n[5], n[6]] });
          break;
        }
        default:
          out.skipped.push(raw);
      }
    } catch {
      out.skipped.push(raw);
    }
  }
  return out;
}

function nums(parts: string[], count: number): number[] {
  if (parts.length !== count) throw new Error(`expected ${count} numbers, got ${parts.length}`);
  const n = parts.map(Number);
  if (n.some((x) => !Number.isFinite(x))) throw new Error('non-numeric field');
  return n;
}
