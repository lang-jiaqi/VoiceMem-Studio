const SIZES = { dot: [52, 52], sit: [370, 510], lie: [580, 440] };
function selectPose(random = Math.random) { return random() < 0.5 ? 'sit' : 'lie'; }
function fitBounds(anchor, size, area) {
  const width = Math.min(size[0], area.width), height = Math.min(size[1], area.height);
  return { width, height,
    x: Math.round(Math.max(area.x, Math.min(anchor.x - width, area.x + area.width - width))),
    y: Math.round(Math.max(area.y, Math.min(anchor.y - height, area.y + area.height - height))) };
}
module.exports = { SIZES, selectPose, fitBounds };
