// Open Referee service worker: installability + offline report caching.
const CACHE = "open-referee-v1";
const OFFLINE_URLS = ["/", "/static/app.css"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(OFFLINE_URLS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  // Network-first for pages & API; cache-first for static assets.
  if (url.pathname.startsWith("/static/")) {
    event.respondWith(
      caches.match(event.request).then(
        (hit) =>
          hit ||
          fetch(event.request).then((resp) => {
            const copy = resp.clone();
            caches.open(CACHE).then((c) => c.put(event.request, copy));
            return resp;
          })
      )
    );
    return;
  }
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/review/")) {
    // never cache live run data; let reports be served by the server when online
    return;
  }
  event.respondWith(
    fetch(event.request)
      .then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE).then((c) => c.put(event.request, copy));
        return resp;
      })
      .catch(() => caches.match(event.request).then((hit) => hit || caches.match("/")))
  );
});
