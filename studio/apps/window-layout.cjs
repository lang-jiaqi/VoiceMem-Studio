'use strict';

/** Return centered Studio bounds and minimum sizes within a display's work area. */
function studioWindowLayout(home, workArea) {
  const preferred = home
    ? { width: 720, height: 560, minWidth: 480, minHeight: 380 }
    : { width: 1440, height: 920, minWidth: 1080, minHeight: 680 };
  const width = Math.min(preferred.width, Math.max(1, workArea.width - 32));
  const height = Math.min(preferred.height, Math.max(1, workArea.height - 32));
  return {
    x: workArea.x + Math.round((workArea.width - width) / 2),
    y: workArea.y + Math.round((workArea.height - height) / 2),
    width, height,
    minWidth: Math.min(preferred.minWidth, width),
    minHeight: Math.min(preferred.minHeight, height),
  };
}

module.exports = { studioWindowLayout };
