import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5710, strictPort: true, proxy: { '/api': 'http://localhost:5711' }, watch: { ignored: ['**/state.json', '**/pipeline.log', '**/raw.ply', '**/aligned.ply', '**/raw-*.glb', '**/*-urdf.zip', '**/rooms-input/**', '**/spike-out/**'] } },
  optimizeDeps: { exclude: ['@mujoco/mujoco', 'onnxruntime-web', '@manycore/aholo-viewer'] },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
