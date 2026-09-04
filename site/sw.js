const CACHE_NAME = 'redfox-shell-v2';
const APP_SHELL = ['/index.html', '/app.html', '/manifest.webmanifest', '/app-icon.png', '/app-icon.svg', '/pwa.js'];

self.addEventListener('install', function (event) {
  event.waitUntil(caches.open(CACHE_NAME).then(function (cache) {
    return cache.addAll(APP_SHELL);
  }));
  self.skipWaiting();
});

self.addEventListener('activate', function (event) {
  event.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (key) { return key !== CACHE_NAME; }).map(function (key) {
      return caches.delete(key);
    }));
  }));
  self.clients.claim();
});

self.addEventListener('fetch', function (event) {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  // Never persist authenticated board data or the board document in the app cache.
  if (url.pathname === '/board.html' || url.pathname.startsWith('/data/') || url.pathname.startsWith('/auth/') || url.pathname.startsWith('/functions/')) return;

  if (request.mode === 'navigate') {
    event.respondWith(fetch(request).catch(function () {
      return caches.match('/index.html');
    }));
    return;
  }

  event.respondWith(caches.match(request).then(function (cached) {
    return cached || fetch(request);
  }));
});
