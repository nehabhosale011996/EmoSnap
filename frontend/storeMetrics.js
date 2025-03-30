
// save metrics to Local Storage
function saveMetricsToLocalStorage() {
    let currentUser = JSON.parse(localStorage.getItem("currentUser"));

    if (!currentUser) {
        console.log("No user data found.");
        return;
    }

    let currentPlayerData = {
        FirstName: currentUser.firstName,
        LastInitials: currentUser.lastInitials,
        Grade: currentUser.grade,
        BestStage: currentUser.fastestStageName || "N/A",
        BestTime: currentUser.fastestStage || "N/A",
        ToughestStage: currentUser.slowestStageName || "N/A",
        ToughestTime: currentUser.slowestStage || "N/A",
        TotalTime: currentUser.totalTime || "N/A",
        CorrectAnswers: currentUser.correctAnswers || 0,
        CurrentStage: currentUser.currentStage + 1  // Include the last played stage
    };

    // Store the current data
    let metricsData = JSON.parse(localStorage.getItem("allPlayersMetrics")) || [];
    metricsData.push(currentPlayerData);
    localStorage.setItem("allPlayersMetrics", JSON.stringify(metricsData));

    console.log("Metrics saved:", metricsData);
}

// Trigger Excel export when game stops
window.addEventListener("beforeunload", () => {
    saveMetricsToLocalStorage();
});


// downloads Metrics into Excel file
function downloadMetricsExcel() {
    let metricsData = JSON.parse(localStorage.getItem("allPlayersMetrics")) || [];

    if (metricsData.length === 0) {
        alert("No metrics data available.");
        return;
    }

    // Convert to worksheet
    let worksheet = XLSX.utils.json_to_sheet(metricsData);

    // Create a workbook
    let workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "MetricsData");

    // Save the file
    XLSX.writeFile(workbook, "EmoSnap_Metrics.xlsx");

}
