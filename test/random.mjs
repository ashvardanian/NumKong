/**
 *  @file test/random.mjs
 *  @author Ash Vardanian
 *  @date September 30, 2026
 *  @brief SplitMix64 for the JavaScript tests and benchmarks, bit-identical to the C++ reference.
 *
 *  Each stream is keyed by `streamKey(seed, name)`, so neither adjacent seeds nor similar names
 *  yield related streams. Never seed a `Random` with `seed + 1`.
 */

const MASK_64 = (1n << 64n) - 1n;

/** Low 64 bits of the product of two 64-bit numbers, each given as 32-bit halves. */
function multiply64(aHigh, aLow, bHigh, bLow) {
  const a0 = aLow & 0xffff, a1 = aLow >>> 16, b0 = bLow & 0xffff, b1 = bLow >>> 16;
  const middle = a1 * b0 + a0 * b1;
  const low = a0 * b0 + (middle & 0xffff) * 0x10000;
  const carry = Math.floor(low / 0x100000000);
  const high = Math.imul(aHigh, bLow) + Math.imul(aLow, bHigh) + a1 * b1 + Math.floor(middle / 0x10000) + carry;
  return [high >>> 0, low >>> 0];
}

/** One `x ^= x >> shift; x *= multiplier` step of the SplitMix64 finalizer, over 32-bit halves. */
function xorShiftMultiply(high, low, shift, multiplierHigh, multiplierLow) {
  low = (low ^ ((low >>> shift) | (high << (32 - shift)))) >>> 0;
  high = (high ^ (high >>> shift)) >>> 0;
  return multiply64(high, low, multiplierHigh, multiplierLow);
}

/** The SplitMix64 finalizer over BigInt, for deriving stream keys. */
function mix64(x) {
  x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK_64;
  x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK_64;
  return x ^ (x >> 31n);
}

/** 64-bit FNV-1a over the UTF-16 code units of `name`, all ASCII in practice. */
function fnv1a64(name) {
  let hash = 0xcbf29ce484222325n;
  for (let i = 0; i < name.length; i++) hash = ((hash ^ BigInt(name.charCodeAt(i))) * 0x100000001b3n) & MASK_64;
  return hash;
}

/** Stream key `mix(mix(seed ^ fnv1a64(name)) + index)`, one per test, lane or purpose. */
export function streamKey(seed, name, index = 0) {
  return mix64((mix64(BigInt(seed) ^ fnv1a64(name)) + BigInt(index)) & MASK_64);
}

/** SplitMix64 over two 32-bit halves, bit-identical to the reference C++ SplitMix64. */
export class Random {
  constructor(key) {
    this.high = Number((key >> 32n) & 0xffffffffn);
    this.low = Number(key & 0xffffffffn);
  }

  /** The next raw 64-bit output as `[high, low]` 32-bit halves. */
  next64() {
    const sum = this.low + 0x7f4a7c15;
    this.low = sum >>> 0;
    this.high = (this.high + 0x9e3779b9 + (sum > 0xffffffff ? 1 : 0)) >>> 0;
    let [high, low] = xorShiftMultiply(this.high, this.low, 30, 0xbf58476d, 0x1ce4e5b9);
    [high, low] = xorShiftMultiply(high, low, 27, 0x94d049bb, 0x133111eb);
    return [(high ^ (high >>> 31)) >>> 0, (low ^ ((low >>> 31) | (high << 1))) >>> 0];
  }

  /** Uniform in [0, 1) from the top 53 bits, like `uniform_between<f64_t>`. */
  next() {
    const [high, low] = this.next64();
    return (high * 0x200000 + (low >>> 11)) * 2 ** -53;
  }
}
