const SIZES = { dot: [52, 52], sit: [480, 640], lie: [480, 640] };
const MIN_SCALE = 0.4, MAX_SCALE = 1.5;
const RESIZE_CORNERS = { nw: [-1, -1], ne: [1, -1], sw: [-1, 1], se: [1, 1] };
function clampScale(value) {
  return Number.isFinite(value) ? Math.min(MAX_SCALE, Math.max(MIN_SCALE, value)) : 1;
}
function scaledSize(mode, scale = 1) {
  return SIZES[mode].map(value => Math.round(value * (mode === 'dot' ? 1 : clampScale(scale))));
}
function draggedScale(scale, dx, dy) {
  const horizontal = dx / SIZES.sit[0], vertical = dy / SIZES.sit[1];
  return clampScale(scale + (Math.abs(horizontal) > Math.abs(vertical) ? horizontal : vertical));
}
function resizeFromCorner(mode, start, dx, dy, area) {
  const [horizontal, vertical] = RESIZE_CORNERS[start.corner];
  const scale = draggedScale(start.scale, dx * horizontal, dy * vertical);
  const size = scaledSize(mode, scale), bounds = start.bounds;
  const anchor = {
    x: bounds.x + (horizontal < 0 ? bounds.width : size[0]),
    y: bounds.y + (vertical < 0 ? bounds.height : size[1]),
  };
  return { scale, bounds: fitBounds(anchor, size, area) };
}
function selectPose(random = Math.random) { return random() < 0.5 ? 'sit' : 'lie'; }
function fitBounds(anchor, size, area) {
  const factor = Math.min(1, area.width / size[0], area.height / size[1]);
  const width = Math.max(1, Math.round(size[0] * factor)), height = Math.max(1, Math.round(size[1] * factor));
  return { width, height,
    x: Math.round(Math.max(area.x, Math.min(anchor.x - width, area.x + area.width - width))),
    y: Math.round(Math.max(area.y, Math.min(anchor.y - height, area.y + area.height - height))) };
}
module.exports = { SIZES, MIN_SCALE, MAX_SCALE, RESIZE_CORNERS, clampScale, scaledSize, draggedScale, resizeFromCorner, selectPose, fitBounds };
