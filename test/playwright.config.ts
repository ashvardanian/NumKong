import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: '.',
    testMatch: ['test-browser-runner.mjs', 'test-browser-bundle-runner.mjs'],
    timeout: 120000,
    reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
    use: {
        baseURL: 'http://localhost:8888',
    },
    // WebKit is the one engine without Relaxed SIMD, so it is the one that loads the strict module.
    projects: [
        { name: 'chromium', use: { browserName: 'chromium' } },
        { name: 'webkit', use: { browserName: 'webkit' } },
    ],
    webServer: {
        command: 'npx http-server .. -p 8888 -c-1 --silent',
        port: 8888,
        reuseExistingServer: !process.env.CI,
    },
});
