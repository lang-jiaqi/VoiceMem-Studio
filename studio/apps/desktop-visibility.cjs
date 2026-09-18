'use strict';

function focused(window) {
  return Boolean(window && !window.isDestroyed() && window.isFocused());
}

function shouldShowPet(petEnabled, studio, launcher) {
  return Boolean(petEnabled && studio && !studio.isDestroyed()
    && !focused(studio) && !focused(launcher));
}

module.exports = { shouldShowPet };
