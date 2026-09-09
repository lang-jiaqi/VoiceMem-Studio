// 关键帧逐个照抄 models/*/noctelle-*.idle.motion3.json——sit 和 lie 两份内容完全一样。
// 6 秒循环，段间线性（原文件就是线性的，不擅自改成缓动）。
//
// 不走 Cubism 的 motionManager 播这个文件，理由跟点头、歪头笑一样：时间线会独占
// 它动的参数，而这里待机要跟说话口型、眨眼、点头同时存在。
//
// 注意作者给的幅度：ParamAngleZ 只有 ±2，量程是 ±30。原样播出来几乎看不出在动，
// 所以留了 scale——只放大角度，呼吸和眼睛是 0~1 的开合量，放大会直接坏掉。
(function (root, factory) {
  const profile = factory();
  if (typeof module === 'object' && module.exports) module.exports = profile;
  else root.Idle = profile;
})(globalThis, () => {
  const duration = 6;
  const CURVES = {
    ParamAngleZ:     [[0, -2], [1.5, 2], [3, -2], [4.5, 2], [6, -2]],
    ParamBodyAngleX: [[0, -1.2], [3, 1.2], [6, -1.2]],
    ParamBreath:     [[0, 0], [1.5, 1], [3, 0], [4.5, 1], [6, 0]],
    ParamEyeLOpen:   [[0, 1], [2.7, 1], [2.78, 0], [2.88, 1], [6, 1]],
    ParamEyeROpen:   [[0, 1], [2.7, 1], [2.78, 0], [2.88, 1], [6, 1]],
  };
  // 只有这两条是角度，scale 只作用在它们身上。
  const SCALABLE = new Set(['ParamAngleZ', 'ParamBodyAngleX']);

  function track(keys, seconds) {
    for (let i = 1; i < keys.length; i++) {
      const [t1, v1] = keys[i];
      if (seconds <= t1) {
        const [t0, v0] = keys[i - 1];
        return t1 === t0 ? v1 : v0 + (v1 - v0) * ((seconds - t0) / (t1 - t0));
      }
    }
    return keys[keys.length - 1][1];
  }

  function sample(seconds, scale = 1) {
    const phase = ((seconds % duration) + duration) % duration;
    const out = {};
    for (const [id, keys] of Object.entries(CURVES)) {
      const value = track(keys, phase);
      out[id] = SCALABLE.has(id) ? value * scale : value;
    }
    return out;
  }

  return { duration, sample, CURVES };
});
