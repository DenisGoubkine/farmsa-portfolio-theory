document.addEventListener("DOMContentLoaded", () => {
  const codeBlocks = Array.from(document.querySelectorAll("details.code-fold"));
  if (codeBlocks.length === 0) return;

  const titleBlock = document.querySelector(".quarto-title-block");
  if (!titleBlock) return;

  const controls = document.createElement("div");
  controls.className = "report-controls";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "code-toggle-button";

  const setState = (open) => {
    for (const block of codeBlocks) {
      block.open = open;
      const summary = block.querySelector("summary");
      if (summary) {
        summary.textContent = open ? "Hide code" : "Show code";
      }
    }
    button.textContent = open ? "Hide All Code" : "Show All Code";
    button.setAttribute("aria-pressed", open ? "true" : "false");
  };

  let allOpen = codeBlocks.every((block) => block.open);
  setState(allOpen);

  button.addEventListener("click", () => {
    allOpen = !allOpen;
    setState(allOpen);
  });

  controls.appendChild(button);
  titleBlock.appendChild(controls);
});
