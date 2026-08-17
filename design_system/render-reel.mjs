import { chromium } from 'playwright';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

const DURATION = 36;
const FPS = 30;
const SUPERSAMPLE = 3;
const WIDTH = 1920 * SUPERSAMPLE;   // 5760
const HEIGHT = 1080 * SUPERSAMPLE;  // 3240
const OUT_WIDTH = 1920;
const OUT_HEIGHT = 1080;
const TOTAL_FRAMES = DURATION * FPS;

const lang = process.argv[2] || 'en';
const inputFile = path.resolve(`ecodb-reel-render-${lang}.html`);
const framesDir = path.resolve(`frames-${lang}`);
const outputFile = path.resolve(`ecodb-reel-${lang}.mp4`);

if (!fs.existsSync(inputFile)) {
  console.error(`Input file not found: ${inputFile}`);
  process.exit(1);
}

fs.mkdirSync(framesDir, { recursive: true });

console.log(`Rendering ${lang.toUpperCase()} — ${TOTAL_FRAMES} frames at ${FPS}fps (${DURATION}s)`);
console.log(`Input:  ${inputFile}`);
console.log(`Output: ${outputFile}`);

const browser = await chromium.launch({ headless: true, args: ['--disable-gpu-sandbox'] });
const page = await browser.newPage({
  viewport: { width: WIDTH, height: HEIGHT },
  deviceScaleFactor: 1
});

await page.goto(`file:///${inputFile.replace(/\\/g, '/')}`, { waitUntil: 'networkidle' });

// Wait for bundler to unpack and React to mount
console.log('Waiting for composition to mount...');
await page.waitForFunction(() => window.__hf_ready === true, { timeout: 60000 });
console.log('Composition ready. Starting frame capture...');

// Hide the playback bar
await page.evaluate(() => {
  const bar = document.querySelector('[class*="playback"], [style*="height: 44px"]');
  if (bar) bar.style.display = 'none';
  // Also hide by finding the bottom flex child of the stage
  const stage = document.querySelector('[style*="flex-direction: column"]');
  if (stage && stage.children.length > 1) {
    const last = stage.children[stage.children.length - 1];
    if (last.querySelector('button') || last.querySelector('svg')) {
      last.style.display = 'none';
    }
  }
});

// Capture frames
const startTime = Date.now();
for (let frame = 0; frame < TOTAL_FRAMES; frame++) {
  const t = frame / FPS;
  await page.evaluate((seekTime) => window.__hf.seek(seekTime), t);
  // Small wait for React to re-render
  await page.waitForTimeout(16);

  const frameNum = String(frame).padStart(5, '0');
  await page.screenshot({
    path: path.join(framesDir, `frame_${frameNum}.png`),
    clip: { x: 0, y: 0, width: WIDTH, height: HEIGHT }
  });

  if (frame % 30 === 0) {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const pct = ((frame / TOTAL_FRAMES) * 100).toFixed(0);
    console.log(`  ${pct}% — frame ${frame}/${TOTAL_FRAMES} (${elapsed}s elapsed)`);
  }
}

console.log('Frame capture complete. Assembling video with ffmpeg...');
await browser.close();

// Assemble with ffmpeg — downscale from 3x to 1080p with Lanczos
const ffmpegCmd = [
  'C:\\Users\\Admin\\ffmpeg\\bin\\ffmpeg.exe',
  '-y',
  '-framerate', String(FPS),
  '-i', path.join(framesDir, 'frame_%05d.png'),
  '-vf', `scale=${OUT_WIDTH}:${OUT_HEIGHT}:flags=lanczos`,
  '-c:v', 'libx264',
  '-preset', 'slow',
  '-crf', '18',
  '-pix_fmt', 'yuv420p',
  '-movflags', '+faststart',
  outputFile
].join(' ');

console.log('Running ffmpeg...');
execSync(ffmpegCmd, { stdio: 'inherit' });

// Cleanup frames
console.log('Cleaning up frames...');
fs.rmSync(framesDir, { recursive: true });

const stats = fs.statSync(outputFile);
console.log(`\nDone! ${outputFile} (${(stats.size / 1024 / 1024).toFixed(1)} MB)`);
