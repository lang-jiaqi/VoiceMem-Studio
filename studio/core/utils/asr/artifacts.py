"""Acquire official X-ASR artifacts and publish a verified local INT8 profile."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from studio.core.utils.models.component import Model
from .profile import (XASR_FILES, XASR_PARTS, XASR_RECIPE, XASR_REPOSITORY,
                      XASR_REVISION, XASR_SOURCE)


def _identity():
    return dict(repository=XASR_REPOSITORY, revision=XASR_REVISION, recipe=XASR_RECIPE)


class XASRModel(Model):
    """Cache local conversion; incomplete downloads or builds remain unready."""

    def problems(self, root):
        problems = super().problems(root)
        directory = Path(root) / self.directory
        try:
            receipt = json.loads((directory / 'prepared.json').read_text())
            if any(receipt.get(key) != value for key, value in _identity().items()):
                problems.append('X-ASR INT8 来源或量化版本不匹配')
            sizes = receipt.get('sizes', {})
            for name in XASR_FILES[:-1]:
                if not (directory / name).is_file() or sizes.get(name) != (directory / name).stat().st_size:
                    problems.append(f'X-ASR INT8 文件未完成：{name}')
        except (OSError, ValueError, AttributeError, TypeError):
            problems.append('缺少有效的 X-ASR INT8 准备记录')
        return list(dict.fromkeys(problems))

    def acquire(self, root, legacy_root):
        """Download a pinned revision, then atomically mark a successful build ready.

        Conversion runs outside the serving process. Failures propagate and the
        next launch reuses completed downloads without accepting partial weights.
        """
        from filelock import FileLock
        from huggingface_hub import hf_hub_download
        root = Path(root)
        target = root / self.directory
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(target) + '.lock'):
            if self.ready(root):
                print(f'[model] 就绪：{self.name}', flush=True)
                return target
            for relative in self.legacy or (self.directory,):
                old = Path(legacy_root) / relative
                probe = XASRModel(self.name, relative, self.repository, self.required)
                if not target.exists() and not target.is_symlink() and probe.ready(legacy_root):
                    target.symlink_to(old.resolve(), target_is_directory=True)
                    print(f'[model] 复用已有权重：{self.name}', flush=True)
                    return target
            target.mkdir(parents=True, exist_ok=True)
            print('[model] X-ASR：下载官方权重并准备本地 INT8 缓存（仅首次需要）', flush=True)
            try:
                source = None
                for name in tuple(f'{part}-480ms.onnx' for part in XASR_PARTS) + ('tokens.txt',):
                    path = hf_hub_download(
                        repo_id=self.repository, revision=XASR_REVISION,
                        filename=f'{XASR_SOURCE}/{name}', local_dir=str(target / '.source'))
                    source = Path(path).parent
                with tempfile.TemporaryDirectory(prefix='.xasr-build-', dir=target.parent) as temporary:
                    build = Path(temporary)
                    subprocess.run([sys.executable, str(Path(__file__).with_name('quantize.py')),
                                    str(source.resolve()), str(build.resolve())],
                                   cwd=build, check=True)
                    shutil.copyfile(source / 'tokens.txt', build / 'tokens.txt')
                    receipt = _identity()
                    receipt['sizes'] = {name: (build / name).stat().st_size for name in XASR_FILES[:-1]}
                    (build / 'prepared.json').write_text(json.dumps(receipt, indent=2) + '\n')
                    probe = XASRModel(self.name, build.name, self.repository, self.required)
                    problems = probe.problems(build.parent)
                    if problems:
                        raise RuntimeError('；'.join(problems))
                    (target / 'prepared.json').unlink(missing_ok=True)
                    for name in XASR_FILES:
                        (build / name).replace(target / name)
            except Exception as exc:
                raise RuntimeError(f'X-ASR INT8 准备失败：{type(exc).__name__}: {exc}；重新启动会复用已下载内容') from exc
            print('[model] X-ASR INT8 已就绪，后续启动直接复用', flush=True)
            return target
