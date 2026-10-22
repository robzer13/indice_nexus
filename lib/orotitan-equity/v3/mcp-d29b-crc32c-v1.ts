/**
 * Pure CRC-32C (Castagnoli) integrity checksum for Google Cloud KMS requests.
 * The reflected polynomial is 0x82F63B78. Never confuse CRC32C with CRC32.
 * Returns a non-negative unsigned 32-bit value, suitable for Google's int64
 * dataCrc32c/signatureCrc32c fields.
 */
export function crc32c(input:Uint8Array):number {
  let checksum=0xffffffff;
  for(const byte of input) {
    checksum^=byte;
    for(let bit=0;bit<8;bit++) {
      checksum=(checksum>>>1)^((checksum&1)?0x82f63b78:0);
    }
  }
  return (checksum^0xffffffff)>>>0;
}
