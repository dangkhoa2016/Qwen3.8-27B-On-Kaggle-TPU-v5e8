const { test, expect } = require('@playwright/test');
const { mkdtempSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const { execFileSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');
let directory;
test.beforeAll(() => {
  directory = mkdtempSync(join(tmpdir(), 'qwen38-synthetic-task-gallery-'));
  const python = [
    'import sys,json,hashlib',
    'from pathlib import Path',
    "sys.path.insert(0, 'tests')",
    'from test_gallery import fixture_results, EXAMPLES, MANIFEST',
    'from gallery.render import build_gallery,sha256',
    'base=Path(' + JSON.stringify(directory) + ')',
    'en,cases=fixture_results(base)',
    "build_gallery(en,EXAMPLES/'images',base/'baseline',manifest=MANIFEST)",
    "eligible=[r for r in cases if r['group']!='ocr']",
    "vi={'source_full19':en.name,'source_full19_sha256':sha256(en),'status':'COMPLETED','cases':[]}",
    'for row in eligible:',
    "    vi['cases'].append({'id':row['id'],'group':row['group'],'prompt':'VI synthetic','generated_text':'SYNTHETIC VI answer','generated_token_ids':[5],'eos_reached':True,'image_sha256':row.get('image_sha256')})",
    "vi_file=base/'vi.json';vi_file.write_text(json.dumps(vi))",
    "build_gallery(en,EXAMPLES/'images',base/'with-vi',vi_file,MANIFEST)",
    "findings=[]",
    'for idx,row in enumerate(eligible):',
    "    findings.append({'id':row['id'],'verdict':('core_equivalent','partial_equivalent','material_divergence')[idx%3],'common':{'en':'SYNTHETIC core','vi':'SYNTHETIC chung'},'reference':{'en':'SYNTHETIC fact','vi':'SYNTHETIC dữ kiện'},'difference':{'en':'SYNTHETIC difference','vi':'SYNTHETIC khác biệt'}})",
    "from collections import Counter",
    "review={'schema':'qwen38-pairwise-review-v1','status':'PRELIMINARY_MANUAL_REVIEW','methodology':{'en':'Synthetic test','vi':'Kiểm thử giả lập'},'source':{'en_sha256':sha256(en),'vi_sha256':sha256(vi_file),'evidence_sha256':sha256(base/'with-vi/evidence.json')},'records':findings,'counts':dict(Counter(x['verdict'] for x in findings))}",
    "file=base/'review.json';file.write_text(json.dumps(review))",
    "build_gallery(en,EXAMPLES/'images',base/'reviewed',vi_file,MANIFEST,file)"
  ].join('\n');
  execFileSync('python3', ['-c', python], { cwd: process.cwd() });
});
test.afterAll(() => { if (directory) rmSync(directory, { recursive: true, force: true }); });
test('Baseline task gallery groups 19 cases without inventing VI or requiring manual review', async ({ page }) => {
  const errors=[];page.on('pageerror', e=>errors.push(String(e)));
  await page.goto(pathToFileURL(join(directory, 'baseline','index.html')).href);
  await expect(page.locator('[data-group]')).toHaveCount(5);
  await expect(page.locator('.case-tile')).toHaveCount(4);
  await expect(page.locator('#detail')).toContainText('Chưa sinh VI');
  await expect(page.locator('#severity')).toBeHidden();
  await expect(page.locator('#reviewLink')).toBeHidden();
  await expect(page.locator('.review-panel')).toHaveCount(0);
  await page.locator('[data-group="extract"]').click();
  await expect(page.locator('.case-tile')).toHaveCount(4);
  await page.locator('[data-group="scene"]').click();
  await expect(page.locator('.case-tile')).toHaveCount(6);
  await page.locator('[data-group="ocr"]').click();
  await expect(page.locator('.case-tile')).toHaveCount(2);
  await expect(page.locator('#detail')).toContainText('không thực hiện suy luận VI');
  await page.locator('[data-group="text"]').click();
  await expect(page.locator('.case-tile')).toHaveCount(3);
  await page.locator('#lang').click();
  await expect(page.locator('#title')).toContainText('Task-based inference gallery');
  await expect(page.locator('#detail')).not.toContainText('<|im_end|>');
  expect(errors).toEqual([]);
});
test('VI output and real semantic overlay are separate independent layers', async ({ page }) => {
  const errors=[];page.on('pageerror', e=>errors.push(String(e)));
  await page.goto(pathToFileURL(join(directory,'with-vi','index.html')).href);
  await expect(page.locator('.review-panel')).toHaveCount(0);
  await expect(page.locator('#detail')).toContainText('SYNTHETIC VI answer');
  await page.goto(pathToFileURL(join(directory,'reviewed','index.html')).href);
  await expect(page.locator('.review-panel')).toContainText('SYNTHETIC khác biệt');
  await page.locator('#severity').selectOption('material_divergence');
  await expect(page.locator('.case-tile').first()).toBeVisible();
  await page.locator('#severity').selectOption('');
  await page.locator('#search').fill('SYNTHETIC fact');
  await expect(page.locator('.case-tile')).toHaveCount(4);
  await page.locator('#search').fill('photo-business-card');
  await expect(page.locator('.case-tile')).toHaveCount(1);
  await expect(page.locator('#detail')).toContainText('SYNTHETIC VI answer');
  expect(errors).toEqual([]);
});
test('Gallery is responsive without horizontal overflow at 390px', async ({ page }) => {
  await page.setViewportSize({width:390,height:844});
  await page.goto(pathToFileURL(join(directory,'reviewed','index.html')).href);
  await expect(page.locator('.case-tile')).toHaveCount(4);
  expect(await page.evaluate(() => document.documentElement.scrollWidth<=window.innerWidth+1)).toBe(true);
  await expect(page.locator('#detail .response')).toHaveCount(2);
});
