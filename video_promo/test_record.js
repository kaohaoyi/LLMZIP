const playwright = require('E:/Dev/AI/html-video-temp/packages/adapter-hyperframes/node_modules/playwright');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

async function test() {
  console.log("Launching Chromium...");
  const tempDir = path.join(__dirname, 'temp_rec');
  if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir, { recursive: true });

  const browser = await playwright.chromium.launch({
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1080, height: 1920 },
    recordVideo: {
      dir: tempDir,
      size: { width: 1080, height: 1920 }
    }
  });

  const page = await context.newPage();
  const fileUrl = 'file:///' + path.resolve(__dirname, 'promo.html').replace(/\\/g, '/') + '?lang=zh-tw';
  console.log("Opening:", fileUrl);
  await page.goto(fileUrl);

  // Wait 3 seconds for test
  await page.waitForTimeout(3000);

  await page.close();
  await context.close();
  await browser.close();

  const files = fs.readdirSync(tempDir);
  console.log("Recorded files:", files);
}

test().catch(err => {
  console.error("Test failed:", err);
  process.exit(1);
});
