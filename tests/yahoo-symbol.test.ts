import assert from 'node:assert/strict';
import test from 'node:test';
import { toYahooFinanceSymbol } from '../lib/market/yahoo-symbol';

test('Yahoo Finance symbols map European exchanges deterministically', () => {
  assert.equal(toYahooFinanceSymbol('AUTO:LSE'), 'AUTO.L');
  assert.equal(toYahooFinanceSymbol('BCG:LSE'), 'BCG.L');
  assert.equal(toYahooFinanceSymbol('RMS:EPA'), 'RMS.PA');
  assert.equal(toYahooFinanceSymbol('MEDI:OSE'), 'MEDI.OL');
  assert.equal(toYahooFinanceSymbol('RAA:XETR'), 'RAA.DE');
  assert.equal(toYahooFinanceSymbol('ADYEN:AMS'), 'ADYEN.AS');
  assert.equal(toYahooFinanceSymbol('BEAN:SWX'), 'BEAN.SW');
  assert.equal(toYahooFinanceSymbol('BN:TSX'), 'BN.TO');
  assert.equal(toYahooFinanceSymbol('TOI:TSXV'), 'TOI.V');
  assert.equal(toYahooFinanceSymbol('NOVO-B:CPH'), 'NOVO-B.CO');
  assert.equal(toYahooFinanceSymbol('2330:TWSE'), '2330.TW');
  assert.equal(toYahooFinanceSymbol('D05:SGX'), 'D05.SI');
  assert.equal(toYahooFinanceSymbol('G24:XETR'), 'G24.DE');
});

test('Yahoo Finance symbols keep bare US tickers unchanged', () => {
  assert.equal(toYahooFinanceSymbol('QLYS'), 'QLYS');
  assert.equal(toYahooFinanceSymbol('SEIC'), 'SEIC');
});

test('Yahoo Finance mapping rejects unknown exchange codes', () => {
  assert.throws(() => toYahooFinanceSymbol('ABC:UNKNOWN'), /Unsupported Yahoo Finance exchange mapping/);
});
