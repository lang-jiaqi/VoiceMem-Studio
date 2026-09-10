"""Studio diagnostics implementation."""

class Diagnostics:
    def _mem_line(self) -> str:
        parts = []
        try:
            import psutil
            p = psutil.Process()
            parts.append(f"rss {p.memory_info().rss / 1073741824:.1f}G")
            sw = psutil.swap_memory()
            parts.append(f"swap {sw.used / 1073741824:.1f}/{sw.total / 1073741824:.0f}G")
        except Exception:
            pass
        try:
            import mlx.core as mx
            parts.append(f"mlx {mx.get_active_memory() / 1073741824:.1f}G 峰 {mx.get_peak_memory() / 1073741824:.1f}G")
        except Exception:
            pass
        return " · ".join(parts)

    def _lat_note(self, total_ms: float) -> str:
        h = self._LAT_HIST
        h["n"] += 1
        if h["first"] is None:
            h["first"] = total_ms
            return "首轮"
        h["rest"].append(total_ms)
        r = sorted(h["rest"])
        return f"第 {h['n']} 轮 · 后续 {len(r)} 轮中位 {r[len(r) // 2]:.0f}ms · 首轮 {h['first']:.0f}ms"
