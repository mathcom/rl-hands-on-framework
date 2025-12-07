import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 5173,
        host: '0.0.0.0',
		allowedHosts: true,
		proxy: {
          '/api': {
            target: 'http://ollama:11434', // Docker 내부의 Ollama 컨테이너 주소
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api/, '/v1'), // /api -> /v1으로 변환
          },
        },
      },
      plugins: [react()],
      define: {
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
