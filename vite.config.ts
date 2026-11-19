import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5710, strictPort: true },
  optimizeDeps: { exclude: ['@mujoco/mujoco', 'onnxruntime-web'] },
  build: { target: 'es2022' },
});
