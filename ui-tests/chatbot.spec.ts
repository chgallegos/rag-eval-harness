/**
 * UI-layer tests (Playwright).
 *
 * The pytest suite scores WHAT the bot says; this layer tests HOW it is
 * delivered: does the answer render, do citations link somewhere real, do
 * error states behave. Point BASE_URL at the deployed chatbot.
 *
 * Run:  npx playwright install && BASE_URL=<your bot url> npx playwright test
 */
import { test, expect } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';

test.describe('chatbot UI', () => {
  test('answers render for an in-scope question', async ({ page }) => {
    await page.goto(BASE_URL);
    await page.getByRole('textbox').fill('How do I reset my password?');
    await page.keyboard.press('Enter');
    // streaming answers arrive incrementally; wait for a stable message
    const answer = page.locator('[data-testid="bot-message"]').last();
    await expect(answer).toContainText(/password/i, { timeout: 15_000 });
  });

  test('citations link to a source', async ({ page }) => {
    await page.goto(BASE_URL);
    await page.getByRole('textbox').fill('What is the refund window?');
    await page.keyboard.press('Enter');
    const citation = page.locator('[data-testid="citation"]').first();
    await expect(citation).toHaveAttribute('href', /.+/, { timeout: 15_000 });
  });

  test('empty input does not send', async ({ page }) => {
    await page.goto(BASE_URL);
    await page.keyboard.press('Enter');
    await expect(page.locator('[data-testid="bot-message"]')).toHaveCount(0);
  });
});
