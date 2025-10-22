/* _src/common/togglestyle.typ */

#import "/_src/mod.typ": css, js
#css.include-file("_assets/css/global.css")

// The button itself
#html.button(class: "theme-toggle", {[🌙 Dark Mode]})

// The JS logic
#js.inline(```js
const toggleBtn = document.querySelector(".theme-toggle");
const root = document.documentElement;

// Restore saved theme
if (localStorage.getItem("theme") === "dark") {
  root.setAttribute("data-theme", "dark");
  toggleBtn.textContent = "☀️ Light Mode";
}

toggleBtn.addEventListener("click", () => {
  if (root.getAttribute("data-theme") === "dark") {
    root.removeAttribute("data-theme");
    localStorage.setItem("theme", "light");
    toggleBtn.textContent = "🌙 Dark Mode";
  } else {
    root.setAttribute("data-theme", "dark");
    localStorage.setItem("theme", "dark");
    toggleBtn.textContent = "☀️ Light Mode";
  }
});
``` )
