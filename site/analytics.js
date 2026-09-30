/* Collect website statistics only on the published domain. */
(() => {
  const hosts = ['false-benchmark-progress.com', 'www.false-benchmark-progress.com'];
  if (!hosts.includes(window.location.hostname)) return;

  const measurementId = 'G-FZT9V5SZRB';
  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };
  window.gtag('js', new Date());
  window.gtag('config', measurementId);

  const tag = document.createElement('script');
  tag.async = true;
  tag.src = 'https://www.googletagmanager.com/gtag/js?id=' + measurementId;
  document.head.appendChild(tag);
})();
