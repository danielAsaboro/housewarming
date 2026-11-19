import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5710, strictPort: true },
  optimizeDeps: { exclude: ['@mujoco/mujoco', 'onnxruntime-web', '@manycore/aholo-viewer'] },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
