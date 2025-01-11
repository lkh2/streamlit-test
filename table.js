function initializeSearch() {
  const searchInput = document.getElementById("table-search");
  const tableRows = document.querySelectorAll("#data-table tbody tr");

  if (!searchInput || !tableRows.length) {
    setTimeout(initializeSearch, 100);
    return;
  }

  searchInput.addEventListener("input", function (e) {
    const searchTerm = e.target.value.toLowerCase().trim();

    tableRows.forEach((row) => {
      // Get both text content and innerHTML for complete search
      const textContent = row.textContent.toLowerCase();
      const innerHTML = row.innerHTML.toLowerCase();

      // Search in both text and HTML content
      if (textContent.includes(searchTerm) || innerHTML.includes(searchTerm)) {
        row.style.display = "";
      } else {
        row.style.display = "none";
      }
    });
  });
}

// Initialize search when DOM loads
document.addEventListener("DOMContentLoaded", initializeSearch);

// Also initialize when component updates
initializeSearch();
