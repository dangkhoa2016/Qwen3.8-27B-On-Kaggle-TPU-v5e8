const { test, expect } = require('@playwright/test');

test('CPU playground shows honest mock state and returns chat reply', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('MOCK BACKEND · NO MODEL LOADED')).toBeVisible();
  await expect(page.locator('.status')).toContainText('mock · model_loaded=false');
  await page.locator('#prompt').fill('Hello via browser');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.locator('pre').first()).toContainText('[MOCK');
  await expect(page.locator('pre').first()).toContainText('Hello via browser');
});

test('CPU playground submits an image and includes clear non-inference disclaimer', async ({ page }) => {
  await page.goto('/');
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/qGMAAAAASUVORK5CYII=', 'base64');
  await page.locator('#image').setInputFiles({name: 'tiny.png', mimeType: 'image/png', buffer: png});
  await expect(page.locator('img[alt="Selected image preview"]')).toBeVisible();
  await page.locator('#visionPrompt').fill('What do you see?');
  await page.getByRole('button', { name: 'Analyze image' }).click();
  await expect(page.locator('pre').last()).toContainText('NOT VISION INFERENCE');
  await expect(page.locator('pre').last()).toContainText('tiny.png');
});
