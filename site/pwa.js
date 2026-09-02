// Register a small app shell only. Market data and board content stay network-only.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', function () {
    navigator.serviceWorker.register('/sw.js').catch(function () {
      // The site remains fully functional when a browser does not support PWAs.
    });
  });
}
