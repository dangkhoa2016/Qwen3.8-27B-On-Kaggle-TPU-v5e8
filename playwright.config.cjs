const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/browser',
  timeout: 30000,
  retries: 0,
  use: {
    baseURL: 'http://127.0.0.1:18090',
    browserName: 'chromium',
    headless: true,
  },
  webServer: {
    command: 'QWEN38_BACKEND=mock python3 -m uvicorn api.app:app --host 127.0.0.1 --port 18090',
    url: 'http://127.0.0.1:18090/health',
    timeout: 30000,
    reuseExistingServer: false
  },
  reporter: [['list']],
});
