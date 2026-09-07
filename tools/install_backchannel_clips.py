"""把试听定稿的附和片段装进缓存，启动时直接读、不再合成。

用法：python3 tools/install_backchannel_clips.py voice/backchannel
目录里 ``OK_<词>.wav``（24k 单声道 PCM16，已切头尾、已归一）会按当前 Breeze 音色
指纹写成 cache/backchannel/<hash>.pcm，style_idx=0。换参考音频或 instruct 指纹就变，
要重跑一次本脚本（或者让启动时重新合成）。
"""
import glob, os, sys, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("VOICEMEM_BREEZE_REF_AUDIO", "voice/noctelle_ref_short.wav")
from voicemem.breeze_tts import BreezeMLXTTS
from harness.backchannel import BackchannelVoice

src = sys.argv[1] if len(sys.argv) > 1 else "voice/backchannel"
bc = BackchannelVoice(BreezeMLXTTS(), lang="zh")      # 不加载模型，只算指纹
n = 0
for f in sorted(glob.glob(os.path.join(src, "OK_*.wav"))):
    token = os.path.basename(f)[3:-4]
    w = wave.open(f); pcm = w.readframes(w.getnframes()); w.close()
    path = bc._path(token, 0); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pcm); n += 1
    print(f"{token:6s} → {path.name}  {len(pcm)/48000*1000:.0f}ms")
print(f"装了 {n} 条，音色 {bc.voice_id}")
