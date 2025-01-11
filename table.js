function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function initializeSearch() {
  const searchInput = document.getElementById("table-search");
  const tableRows = document.querySelectorAll("#data-table tbody tr");

  if (!searchInput || !tableRows.length) {
    setTimeout(initializeSearch, 100);
    return;
  }

  const performSearch = (searchTerm) => {
    try {
      const regex = new RegExp(escapeRegex(searchTerm), "i");

      tableRows.forEach((row) => {
        const cells = Array.from(row.getElementsByTagName("td"));
        const found = cells.some((cell) => {
          const text = cell.textContent || cell.innerText;
          return regex.test(text);
        });

        row.style.display = found ? "" : "none";
      });
    } catch (e) {
      console.error("Search error:", e);
    }
  };

  const debouncedSearch = debounce((e) => {
    const searchTerm = e.target.value.trim();
    performSearch(searchTerm);
  }, 300);

  searchInput.addEventListener("input", debouncedSearch);
}

// Initialize search when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeSearch);
} else {
  initializeSearch();
}

// Reinitialize on Streamlit updates
const observer = new MutationObserver(() => {
  initializeSearch();
});

observer.observe(document.body, {
  childList: true,
  subtree: true,
});
