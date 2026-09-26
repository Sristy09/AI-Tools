// Theme handling — persists choice in localStorage, respects system preference on first visit
(function () {
  const root = document.documentElement;
  const stored = localStorage.getItem('theme');
  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const initial = stored || (systemDark ? 'dark' : 'light');
  root.setAttribute('data-theme', initial);

  document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('theme-toggle');
    const icon = document.getElementById('theme-icon');
    const label = document.getElementById('theme-label');

    function paint(theme) {
      if (!icon || !label) return;
      if (theme === 'dark') {
        icon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" fill="currentColor"/>';
        label.textContent = 'Dark mode';
      } else {
        icon.innerHTML = '<circle cx="12" cy="12" r="4" fill="currentColor"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>';
        label.textContent = 'Light mode';
      }
    }

    paint(root.getAttribute('data-theme'));

    if (toggle) {
      toggle.addEventListener('click', () => {
        const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        paint(next);
        window.dispatchEvent(new CustomEvent('themechange', { detail: next }));
      });
    }

    // Mobile nav
    const menuBtn = document.getElementById('menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const scrim = document.getElementById('scrim');

    function closeNav() {
      sidebar && sidebar.classList.remove('open');
      scrim && scrim.classList.remove('open');
    }

    if (menuBtn && sidebar) {
      menuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        scrim && scrim.classList.toggle('open');
      });
    }
    if (scrim) scrim.addEventListener('click', closeNav);
  });
})();

// Shared chart color helpers, read live from CSS variables so charts follow the active theme
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartPalette() {
  return {
    text: cssVar('--text-muted'),
    grid: cssVar('--border'),
    human: cssVar('--accent-human'),
    ai: cssVar('--accent-ai'),
    aiLight: cssVar('--accent-ai-light'),
    success: cssVar('--success'),
    danger: cssVar('--danger'),
    surface: cssVar('--surface'),
    series: [
      cssVar('--accent-ai'), cssVar('--accent-human'), cssVar('--success'),
      cssVar('--danger'), cssVar('--accent-ai-light'), cssVar('--text-faint'),
    ],
  };
}

Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
