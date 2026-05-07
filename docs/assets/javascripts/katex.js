const offerBotRenderMath = (root) => {
  if (typeof renderMathInElement !== "function") {
    return;
  }

  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false }
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    throwOnError: false
  });
};

if (typeof document$ !== "undefined") {
  document$.subscribe(({ body }) => offerBotRenderMath(body));
} else {
  document.addEventListener("DOMContentLoaded", () => offerBotRenderMath(document.body));
}
