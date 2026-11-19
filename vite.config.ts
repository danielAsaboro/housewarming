import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5710, strictPort: true, proxy: { '/api': 'http://localhost:5711' }, watch: { ignored: ['**/public/rooms/**', '**/rooms-input/**', '**/spike-out/**'] } },
  optimizeDeps: { exclude: ['@mujoco/mujoco', 'onnxruntime-web', '@manycore/aholo-viewer'] },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
