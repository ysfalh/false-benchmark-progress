/* Citations remain selectable when the clipboard API is unavailable. */
document.querySelectorAll('[data-copy]').forEach((button) => {
  button.hidden = false;
  button.addEventListener('click', async () => {
    const code = document.getElementById(button.dataset.copy);
    const status = button.parentElement.querySelector('[role="status"]');
    try {
      await navigator.clipboard.writeText(code.textContent);
      status.textContent = 'Copied';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Text selected; press Ctrl+C or ⌘C.';
    }
  });
});

// Reveal a supporting section when following a direct link into it.
function revealSupportingAnchor() {
  let target;
  try { target = document.getElementById(decodeURIComponent(location.hash.slice(1))); }
  catch { return; }
  if (!target) return;
  let nested = false;
  for (let node = target; node; node = node.parentElement) {
    if (node.tagName === 'DETAILS') { node.open = true; nested = true; }
  }
  if (nested) requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
}
window.addEventListener('hashchange', revealSupportingAnchor);
window.addEventListener('load', revealSupportingAnchor);
revealSupportingAnchor();
// Match each margin note to its callout's line after fonts and layout settle.
function alignMarginNotes() {
  document.querySelectorAll('[role="doc-noteref"]').forEach((reference) => {
    const note = document.getElementById(reference.getAttribute('href').slice(1));
    const wrapper = note?.closest('.summary-with-note, .section-with-note');
    if (!wrapper) return;
    const callout = reference.closest('sup') || reference;
    const top = callout.getBoundingClientRect().top - wrapper.getBoundingClientRect().top;
    note.style.setProperty('--sidenote-top', `${top}px`);
  });
}
window.addEventListener('resize', alignMarginNotes);
window.addEventListener('load', alignMarginNotes);
if (document.fonts) document.fonts.ready.then(alignMarginNotes);
alignMarginNotes();
