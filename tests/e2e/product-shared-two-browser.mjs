import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const root = 'http://127.0.0.1:3184';
const companyId = '00000000-0000-4000-8000-000000000091';
const password = process.env.ADMIN_PASSWORD;
if (process.env.CI !== 'true' ||
  process.env.OROTITAN_PRODUCT_BROWSER_FIXTURE !== '1' ||
  !password || password.length < 20) {
  throw new Error('PRODUCT_BROWSER_ACCEPTANCE_CI_ONLY');
}

const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
try {
  async function login(context) {
    const page = await context.newPage();
    await page.goto(root + '/admin', { waitUntil: 'domcontentloaded' });
    await page.locator('input[name="password"]').fill(password);
    await page.getByRole('button', { name: 'Se connecter' }).click();
    await page.waitForURL(root + '/admin', { timeout: 30000 });
    const cookies = await context.cookies();
    assert.ok(cookies.some(c => c.name === 'orotitan_admin_session' && c.httpOnly),
      'Real login must create signed HTTP-only admin session');
    await page.goto(root + '/admin/research-requests', { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: 'Boîte de réception Research' }).waitFor();
    return page;
  }
  const phone = await browser.newContext({
    viewport: { width: 390, height: 844 }, isMobile: true,
    deviceScaleFactor: 2,
  });
  const desktop = await browser.newContext({ viewport: { width: 1420, height: 960 } });
  const anonymous = await browser.newContext();
  const anonymousPage = await anonymous.newPage();
  await anonymousPage.goto(root + '/admin/research-requests');
  await anonymousPage.locator('input[name="password"]').waitFor();
  assert.ok(anonymousPage.url().includes('/admin'),
    'Unauthenticated browser must be redirected');

  const phonePage = await login(phone);
  const desktopPage = await login(desktop);
  for (const page of [phonePage, desktopPage]) {
    assert.match(await page.locator('body').innerText(), /partagé non-production/i);
    await page.locator('select[name="company_id"]').selectOption(companyId);
  }
  await phonePage.getByRole('button', { name: /Enregistrer une demande partagée/i }).click();
  await phonePage.waitForURL(/\/admin\/research-requests\?request=/);
  const first = new URL(phonePage.url()).searchParams;
  const requestId = first.get('request');
  assert.match(requestId || '', /^[0-9a-f-]{36}$/i);
  assert.equal(first.get('result'), 'created');
  assert.match(await phonePage.locator('body').innerText(), /REQUESTED_NOT_EXECUTED/);

  // The second independently logged-in browser observes the same durable
  // receipt via the PostgreSQL-backed REST fixture, not shared cookies.
  await desktopPage.reload();
  const desktopText = await desktopPage.locator('body').innerText();
  assert.ok(desktopText.includes(requestId));
  assert.match(desktopText, /Synthetic Research Alpha/);

  await desktopPage.locator('select[name="company_id"]').selectOption(companyId);
  await desktopPage.getByRole('button', { name: /Enregistrer une demande partagée/i }).click();
  await desktopPage.waitForURL(/\/admin\/research-requests\?request=/);
  const second = new URL(desktopPage.url()).searchParams;
  assert.equal(second.get('request'), requestId,
    'Both browser sessions must return identical persistent request UUID');
  assert.equal(second.get('result'), 'existing');

  await phonePage.reload();
  const matching = await phonePage.locator('tbody tr').count();
  assert.equal(matching, 1, 'Parallel browser sessions must not create a second request');
  assert.match(await phonePage.locator('body').innerText(), /aucun Registry run/i);

  console.log('PRODUCT TWO BROWSER PASS: logged-in mobile + desktop, identical shared Research request, dedupe and no execution.');
  await Promise.all([phone.close(), desktop.close(), anonymous.close()]);
} finally {
  await browser.close();
}
