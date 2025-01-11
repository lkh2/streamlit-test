document.addEventListener("DOMContentLoaded", function () {
  const searchInput = document.getElementById("table-search");
  const rows = document.querySelectorAll(".table-row");

  searchInput.addEventListener("input", function (e) {
    const searchTerm = e.target.value.toLowerCase();

    rows.forEach((row) => {
      const text = row.textContent.toLowerCase();
      if (text.includes(searchTerm)) {
        row.classList.remove("hidden");
      } else {
        row.classList.add("hidden");
      }
    });
  });
});
