// onnxruntime-web loads its wasm at runtime; serve the CPU build from /ort/.
import { cpSync, mkdirSync } from 'node:fs';
const src = 'node_modules/onnxruntime-web/dist/';
mkdirSync('public/ort', { recursive: true });
for (const f of ['ort-wasm-simd-threaded.wasm']) cpSync(src + f, 'public/ort/' + f);
