// 关键帧照抄 models/*/noctelle-*.nod.motion3.json（PSD2Live 生成的那份）：
// 低头 18° → 回弹 6° → 停，身体跟约五分之一，低到底时眼睛眯一下，全长 2 秒。
//
// 不走 Cubism 的 motionManager 播那个文件，原因跟歪头笑一样：时间线会独占它动的
// 参数，而这里点头要跟待机摆动、眨眼、说话口型同时存在（"嗯"一声的时候可能正好
// 在眨眼）。所以照 tilted-smile.js 的路子重写成可叠加的采样函数。
(function (root, factory) {
  const profile = factory();
  if (typeof module === 'object' && module.exports) module.exports = profile;
  else root.Nod = profile;
})(globalThis, () => {
  const duration = 2;
  const smoother = x => { const t = Math.max(0, Math.min(1, x)); return t * t * t * (t * (t * 6 - 15) + 10); };

  // 原文件段间是线性的，这里换成 smoothstep：线性在"低到底"那一帧会有个可见的折角。
  const HEAD = [[0, 0], [.55, -18], [1.25, 6], [2, 0]];
  const BODY = [[0, 0], [.55, -4], [1.25, 1.5], [2, 0]];
  const EYES = [[0, 1], [.55, .75], [1.25, 1], [2, 1]];

  function track(keys, seconds) {
    for (let i = 1; i < keys.length; i++) {
      const [t1, v1] = keys[i];
      if (seconds <= t1) {
        const [t0, v0] = keys[i - 1];
        return v0 + (v1 - v0) * smoother((seconds - t0) / (t1 - t0));
      }
    }
    return keys[keys.length - 1][1];
  }

  const running = seconds => Number.isFinite(seconds) && seconds >= 0 && seconds < duration;

  function sample(seconds) {
    if (!running(seconds)) return { active: false, ParamAngleY: 0, ParamBodyAngleY: 0, eyeScale: 1 };
    return {
      active: true,
      ParamAngleY: track(HEAD, seconds),
      ParamBodyAngleY: track(BODY, seconds),
      eyeScale: track(EYES, seconds),
    };
  }

  // 头部是**加**上去的：点头骑在待机的小幅摆动上面，而不是把它顶掉。
  // 眼睛是**乘**的：眨眼和歪头笑各自压过的结果要保留，三者叠起来才不会互相抹掉。
  // 跟 TiltedSmile.compose 一样原地改 values，且必须在 override 之前调用。
  function apply(values, seconds) {
    const nod = sample(seconds);
    if (!nod.active) return values;
    values.ParamAngleY += nod.ParamAngleY;
    values.ParamBodyAngleY += nod.ParamBodyAngleY;
    values.ParamEyeLOpen *= nod.eyeScale;
    values.ParamEyeROpen *= nod.eyeScale;
    return values;
  }

  return { duration, sample, apply };
});
