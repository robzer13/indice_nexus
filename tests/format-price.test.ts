import assert from 'node:assert/strict';
import test from 'node:test';
import { formatPrice } from '../lib/domain/format-price';

test('formats EUR major units', () => {
  assert.equal(formatPrice({ value: 632, currency: 'EUR', quoteUnit: 'MAJOR', priceDecimals: 2 }), '632,00 €');
});

test('formats USD major units', () => {
  assert.equal(formatPrice({ value: 108.88, currency: 'USD', quoteUnit: 'MAJOR', priceDecimals: 2 }), '$108,88');
});

test('formats GBP minor units as pence without converting to pounds', () => {
  assert.equal(formatPrice({ value: 528, currency: 'GBP', quoteUnit: 'MINOR', priceDecimals: 0 }), '528p');
});

test('formats NOK major units', () => {
  assert.equal(formatPrice({ value: 250, currency: 'NOK', quoteUnit: 'MAJOR', priceDecimals: 0 }), '250 NOK');
});

test('null price is not rendered as zero', () => {
  assert.equal(formatPrice({ value: null, currency: 'EUR', quoteUnit: 'MAJOR', priceDecimals: 2 }), 'Non disponible');
});
