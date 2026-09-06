import assert from 'node:assert/strict';
import test from 'node:test';
import { parseTwelveDataInstrumentRef } from '../lib/market/twelve-data-symbol';

test('Twelve Data reference parser separates exchange from ticker', () => {
  assert.deepEqual(parseTwelveDataInstrumentRef('AUTO:LSE'), { symbol: 'AUTO', exchange: 'LSE' });
  assert.deepEqual(parseTwelveDataInstrumentRef('RAA:XETR'), { symbol: 'RAA', exchange: 'XETR' });
  assert.deepEqual(parseTwelveDataInstrumentRef('MEDI:OSE'), { symbol: 'MEDI', exchange: 'OSE' });
});

test('Twelve Data reference parser preserves US ticker without exchange', () => {
  assert.deepEqual(parseTwelveDataInstrumentRef('QLYS'), { symbol: 'QLYS', exchange: null });
});

test('Twelve Data reference parser normalizes Euronext Paris alias', () => {
  assert.deepEqual(parseTwelveDataInstrumentRef('RMS:EPA'), { symbol: 'RMS', exchange: 'EURONEXT' });
});
