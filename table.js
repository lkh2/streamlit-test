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
  console.log("Initializing search functionality...");
  const searchInput = document.getElementById("table-search");
  const tableRows = document.querySelectorAll("#data-table tbody tr");

  if (!searchInput || !tableRows.length) {
    console.log("Elements not found, retrying...");
    setTimeout(initializeSearch, 100);
    return;
  }

  console.log(`Found ${tableRows.length} rows in table`);

  const performSearch = (searchTerm) => {
    try {
      console.log(`Performing search for: "${searchTerm}"`);
      const regex = new RegExp(escapeRegex(searchTerm), "i");
      let matchCount = 0;

      tableRows.forEach((row) => {
        const cells = Array.from(row.getElementsByTagName("td"));
        const found = cells.some((cell) => {
          const text = cell.textContent || cell.innerText;
          return regex.test(text);
        });

        if (found) matchCount++;
        row.style.display = found ? "" : "none";
      });

      console.log(`Search complete. Found ${matchCount} matches`);
    } catch (e) {
      console.error("Search error:", e);
    }
  };

  const debouncedSearch = debounce((e) => {
    const searchTerm = e.target.value.trim();
    performSearch(searchTerm);
  }, 300);

  searchInput.addEventListener("input", debouncedSearch);

  console.log("Search initialization complete");
}

console.log("Script loaded");

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
