/**
 *  @file test/browser-runner.mjs
 *  @author Ash Vardanian
 *  @date February 10, 2026
 *  @brief Playwright test runner for NumKong browser WASM tests.
 *
 *  Launches the browser, runs the tests, and reports the results.
 */

import { test, expect } from "@playwright/test";

const htmlPath = "http://localhost:8888/test/browser.html";

test("NumKong WASM browser tests", async ({ page }) => {
  // Listen for console messages and errors
  page.on("console", (msg) => console.log(`Browser: ${msg.text()}`));
  page.on("pageerror", (err) => console.error(`Page error: ${err.message}`));

  // Navigate to test page
  await page.goto(htmlPath);

  // Wait for tests to complete, when `window.testResults` is set
  await page.waitForFunction(() => window.testResults, { timeout: 60000 });

  // Get test results from page
  const results = await page.evaluate(() => window.testResults);
  const allPassed = await page.evaluate(() => window.testsPassed);

  // Log results
  console.log(`\nBrowser test results:`);
  results.forEach((r) => {
    const icon = r.pass ? "✓" : "✗";
    const msg = r.pass ? r.name : `${r.name} - ${r.error}`;
    console.log(`  ${icon} ${msg}`);
  });

  const passed = results.filter((r) => r.pass).length;
  console.log(`\n${passed}/${results.length} tests passed\n`);

  // Assert all tests passed
  expect(allPassed).toBe(true);

  // Individual assertions for better error reporting
  results.forEach((r) => {
    expect(r.pass, `Test "${r.name}" should pass${r.error ? ": " + r.error : ""}`).toBe(true);
  });
});
