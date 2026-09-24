/**
 *  @file test/browser-bundle-runner.mjs
 *  @author Ash Vardanian
 *  @date February 10, 2026
 *  @brief Playwright test runner for the self-contained NumKong browser WASM bundle.
 *
 *  Validates that the auto-initializing bundle works without a manual `initWasm()` call.
 */

import { test, expect } from "@playwright/test";

const htmlPath = "http://localhost:8888/test/browser-bundle.html";

test("NumKong WASM browser bundle tests", async ({ page }) => {
  page.on("console", (msg) => console.log(`Browser: ${msg.text()}`));
  page.on("pageerror", (err) => console.error(`Page error: ${err.message}`));

  await page.goto(htmlPath);

  // Wait for tests to complete
  await page.waitForFunction(() => window.testResults, { timeout: 60000 });

  const results = await page.evaluate(() => window.testResults);
  const allPassed = await page.evaluate(() => window.testsPassed);

  console.log(`\nBrowser bundle test results:`);
  results.forEach((r) => {
    const icon = r.pass ? "\u2713" : "\u2717";
    const msg = r.pass ? r.name : `${r.name} - ${r.error}`;
    console.log(`  ${icon} ${msg}`);
  });

  const passed = results.filter((r) => r.pass).length;
  console.log(`\n${passed}/${results.length} tests passed\n`);

  expect(allPassed).toBe(true);
  results.forEach((r) => {
    expect(r.pass, `Test "${r.name}" should pass${r.error ? ": " + r.error : ""}`).toBe(true);
  });
});
