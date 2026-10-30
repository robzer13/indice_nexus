import assert from 'node:assert/strict';
import {chromium} from 'playwright';
const root='http://127.0.0.1:3184';
if(process.env.CI!=='true' || process.env.NODE_ENV!=='development' ||
   process.env.OROTITAN_PRODUCT_BROWSER_FIXTURE!=='1' ||
   process.env.OROTITAN_RECOVERY_E2E_FIXTURE!=='1' ||
   !process.env.ADMIN_PASSWORD ||
   !process.env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY){
  throw Error('RECOVERY_BROWSER_TEST_NOT_ISOLATED');
}
const browser=await chromium.launch({headless:true,channel:'chrome',args:['--no-sandbox']});
try{
  const anonymous=await browser.newContext();
  const anonymousPage=await anonymous.newPage();
  await anonymousPage.goto(root+'/admin/research-recovery');
  await anonymousPage.locator('input[name="password"]').waitFor();
  assert.ok(anonymousPage.url().endsWith('/admin'),'anonymous must not see recovery journal');

  const context=await browser.newContext({viewport:{width:1360,height:900}});
  const page=await context.newPage();
  await page.goto(root+'/admin');
  await page.locator('input[name="password"]').fill(process.env.ADMIN_PASSWORD);
  await page.getByRole('button',{name:'Se connecter'}).click();
  await page.waitForURL(root+'/admin');
  const adminCookies=await context.cookies();
  assert.ok(adminCookies.some(c=>c.name==='orotitan_admin_session'&&c.httpOnly));

  await page.goto(root+'/admin/research-recovery?source=asml-observation');
  await page.getByRole('heading',{name:'Research Recovery Loop V1'}).waitFor();
  assert.match(await page.locator('body').innerText(),/ASML · observation miroir/);
  assert.match(await page.locator('body').innerText(),/Tentatives persistées\s*0/);
  assert.equal(await page.getByText('Non encore étudié par le loop.').count(),4);
  const execute=page.getByRole('button',{name:'Lancer / reprendre un cycle borné'});
  await execute.click();
  await page.waitForURL(/\/admin\/research-recovery\?.*result=progressed/,{timeout:60000});
  await page.getByText('REUSE_PRIMARY_ARTIFACTS · CANDIDATE_RECORDED').waitFor();
  const text=await page.locator('body').innerText();
  assert.match(text,/Tentatives persistées\s*4/);
  assert.match(text,/Blocages observés\s*4/);
  assert.match(text,/application\/pdf/);
  assert.match(text,/sha256:[0-9a-f]{64}/);
  assert.match(text,/storage:\/\/orotitan-source-files-v1\/research-recovery-v1/);
  assert.match(text,/G001.*aucun calcul certifié|G001 : impossible/s);
  assert.match(text,/G006 : cours de clôture non ajusté/);
  assert.equal((text.match(/REQUIRES_AUTHORITY/g)??[]).length,3);

  const stateUrl='http://127.0.0.1:48761/__fixture/recovery-state';
  const getState=async()=>{
    const response=await fetch(stateUrl,{headers:{
      apikey:process.env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY,
    }});
    assert.equal(response.status,200);
    return response.json();
  };
  const first=await getState();
  assert.equal(first.objectCount,1);
  assert.equal(first.uploads,1);
  assert.equal(first.objects.length,1);
  assert.match(first.objects[0].path,/^[\w/-]+\/[a-f0-9]{64}\.pdf$/);
  assert.ok(first.objects[0].size>10000);

  await page.getByRole('button',{name:'Lancer / reprendre un cycle borné'}).click();
  await page.waitForURL(/\/admin\/research-recovery\?.*result=no_new_automated_steps/,{timeout:60000});
  assert.match(await page.locator('body').innerText(),/Tentatives persistées\s*4/);
  const second=await getState();
  assert.deepEqual(second,first,'rerunning loop must not duplicate or replace archived bytes');
  console.log('RECOVERY FULL ADMIN BROWSER E2E PASS: synthetic PDF, private create-only storage, PG17 journal 4, repeat idempotent, no Registry writes.');
  await context.close();
  await anonymous.close();
}finally{await browser.close();}
