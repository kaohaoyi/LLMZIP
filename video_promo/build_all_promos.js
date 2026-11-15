const playwright = require('E:/Dev/AI/html-video-temp/packages/adapter-hyperframes/node_modules/playwright');
const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

const langs = [
  { code: 'zh-tw', name: '繁體中文 (Traditional Chinese)' },
  { code: 'zh-cn', name: '簡體中文 (Simplified Chinese)' },
  { code: 'en',    name: '英文 (English)' },
  { code: 'ja',    name: '日文 (Japanese)' }
];

const promoDir = __dirname;
const audioDir = path.join(promoDir, 'audio');
const outputDir = path.join(promoDir, 'output');
const tempDir = path.join(promoDir, 'temp_build');

if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir, { recursive: true });

function getDuration(filePath) {
  const res = spawnSync('ffprobe', [
    '-v', 'error',
    '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1',
    filePath
  ], { encoding: 'utf-8' });
  return parseFloat(res.stdout.trim()) || 40.0;
}

async function renderLang(browser, item) {
  const { code, name } = item;
  console.log(`\n========================================`);
  console.log(`▶ Starting editorial render for: [${code}] ${name}`);
  console.log(`========================================`);

  const voPath = path.join(audioDir, `vo_${code}.mp3`);
  const bgmPath = path.join(audioDir, 'bgm_focus.mp3');
  const finalMp4 = path.join(outputDir, `llmzip_promo_${code}.mp4`);

  if (!fs.existsSync(voPath)) {
    throw new Error(`Voiceover not found: ${voPath}`);
  }

  const voDuration = getDuration(voPath);
  const totalDuration = Math.max(36.0, voDuration + 0.8);
  console.log(`VO Duration: ${voDuration.toFixed(2)}s | Total Video Duration: ${totalDuration.toFixed(2)}s`);

  // Clean temp folder
  fs.readdirSync(tempDir).forEach(f => fs.unlinkSync(path.join(tempDir, f)));

  // 1. Record WebM via Playwright
  console.log(`[1/3] Recording 1080x1920 Editorial Canvas...`);
  const context = await browser.newContext({
    viewport: { width: 1080, height: 1920 },
    recordVideo: {
      dir: tempDir,
      size: { width: 1080, height: 1920 }
    }
  });

  const page = await context.newPage();
  const fileUrl = 'file:///' + path.resolve(promoDir, 'promo_v2.html').replace(/\\/g, '/') + `?lang=${code}`;
  await page.goto(fileUrl);

  // Wait full duration
  await page.waitForTimeout(totalDuration * 1000);

  await page.close();
  await context.close();

  const webmFiles = fs.readdirSync(tempDir).filter(f => f.endsWith('.webm'));
  if (webmFiles.length === 0) {
    throw new Error(`Failed to record WebM for ${code}`);
  }
  const rawWebmPath = path.join(tempDir, webmFiles[0]);
  console.log(`[2/3] Raw recording finished: ${rawWebmPath}`);

  // 2. Transcode & Mux with Natural Voiceover + Relaxing Lo-Fi BGM via FFmpeg
  console.log(`[3/3] Mixing video + natural VO + ambient Lo-Fi BGM via FFmpeg...`);
  
  const ffmpegArgs = [
    '-y',
    '-i', rawWebmPath,
    '-i', voPath,
    '-i', bgmPath,
    '-filter_complex',
    '[1:a]volume=1.0[vo];[2:a]volume=0.18[bgm];[vo][bgm]amix=inputs=2:duration=first:dropout_transition=3[aout]',
    '-map', '0:v',
    '-map', '[aout]',
    '-c:v', 'libx264',
    '-preset', 'fast',
    '-crf', '18',
    '-pix_fmt', 'yuv420p',
    '-r', '30',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-t', totalDuration.toString(),
    finalMp4
  ];

  const ffmpegRes = spawnSync('ffmpeg', ffmpegArgs, { encoding: 'utf-8' });
  if (ffmpegRes.status !== 0) {
    console.error(`FFmpeg error for ${code}:`, ffmpegRes.stderr);
    throw new Error(`FFmpeg transcode failed for ${code}`);
  }

  const outStat = fs.statSync(finalMp4);
  console.log(`✔ SUCCESS: Generated ${finalMp4} (${(outStat.size / (1024 * 1024)).toFixed(2)} MB)`);
}

async function main() {
  console.log("Launching headless Chromium...");
  const browser = await playwright.chromium.launch({ headless: true });

  for (const item of langs) {
    await renderLang(browser, item);
  }

  await browser.close();
  console.log("\n========================================================");
  console.log("🎉 ALL 4 REFINED EDITORIAL PROMO VIDEOS GENERATED!");
  console.log("========================================================");
}

main().catch(err => {
  console.error("Build failed:", err);
  process.exit(1);
});
