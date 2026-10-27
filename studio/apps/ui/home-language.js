/* The first-page choice is shared by both visual styles and new connections. */
(() => {
  const select = document.getElementById('baseLanguage');
  if (!select) return;
  select.value = window.VMSettings.language;
  select.addEventListener('change', () => window.VMSettings.setLanguage(select.value));
})();
