import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        allowedHosts: true,
        proxy: {
          '/api': {
            // [수정] 이제 Ollama가 아니라 Backend 컨테이너로 연결합니다.
            target: 'http://backend:8000', 
            changeOrigin: true,
            // [수정] /api/chat -> /chat 으로 변환하여 전달
            rewrite: (path) => path.replace(/^\/api/, ''), 
          },
        },
      },
      plugins: [react()],
      define: {
        // 불필요한 환경변수 정의 제거됨
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, 'src'), // src 경로로 수정
        }
      }
    };
});