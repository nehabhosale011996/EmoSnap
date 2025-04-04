// app.js

const startBtn = document.getElementById("start-btn");
const permissionPrompt = document.getElementById("permission-prompt");
const cameraContainer = document.getElementById("camera-container");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const stageTitle = document.getElementById("stage-title");
const narrationText = document.getElementById("narration");
const timerDisplay = document.getElementById("timer");
const captureBtn = document.getElementById("capture-btn");

let faceMesh;
let latestLandmarks = null;
let currentStage = 0;
let timeLeft = 20;
let timerInterval;
let currentUser = null;
let dpr = 1; // Default value

// Emojis mapping
const emotionEmojis = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "neutral": "😐",
    "disgust": "🤢",
    "fear": "😨"
};

function loadUser() {
    currentUser = JSON.parse(localStorage.getItem("currentUser"));
}

function startGame() {
    loadUser();
    if (!currentUser) {
        alert("No user found! Please register first.");
        window.location.href = "/";
        return;
    }
    const stageTitles = stages.map(stage => stage.title);
    currentUser.stageNames = stageTitles;
    loadStage(currentStage);
}

startBtn.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        permissionPrompt.style.display = "none";
        cameraContainer.style.display = "block";
        video.onloadeddata = startFaceTracking;
        startGame();
    } catch (err) {
        console.error("Camera access denied!", err);
    }
});

function loadStage(stageIndex) {
    if (stageIndex >= stages.length) {
        endGame();
        saveMetricsToLocalStorage();
        showEndGameConfirmationAlert();
        // stopDetection();
        return;
    }

    const stage = stages[stageIndex];
    stageTitle.textContent = stage.title;
    narrationText.textContent = stage.narration;
    timeLeft = 20;
    startTimer();
}

function startTimer() {
    clearInterval(timerInterval);
    timerInterval = setInterval(() => {
        timeLeft--;
        timerDisplay.textContent = `${timeLeft}s remaining`;

        if (timeLeft <= 0) {
            clearInterval(timerInterval);
            nextStage();
        }
    }, 1000);
}

// Save progress after each stage
function saveProgress() {
    if (currentUser) {
        currentUser.currentStage = currentStage+1;
        currentUser.stageTimes = currentUser.stageTimes || [];
        currentUser.totalTime = currentUser.stageTimes.reduce((a, b) => a + b, 0);
        localStorage.setItem("currentUser", JSON.stringify(currentUser));
    }
}

// Trigger save when the user leaves the game
window.addEventListener("beforeunload", () => {
    saveProgress();
});

function nextStage() {
    if (!currentUser) return;
    currentUser.stageTimes.push(20 - timeLeft);
    currentStage++;
    loadStage(currentStage);
}

function endGame() {
    if (!currentUser) return;
    currentUser.totalTime = currentUser.stageTimes.reduce((a, b) => a + b, 0);
    currentUser.fastestStage = Math.min(...currentUser.stageTimes);
    currentUser.slowestStage = Math.max(...currentUser.stageTimes);

    let fastestStageIndex = currentUser.stageTimes.indexOf(currentUser.fastestStage);
    let slowestStageIndex = currentUser.stageTimes.indexOf(currentUser.slowestStage);

    currentUser.fastestStageName = currentUser.stageNames[fastestStageIndex];
    currentUser.slowestStageName = currentUser.stageNames[slowestStageIndex];

    localStorage.setItem("currentUser", JSON.stringify(currentUser));
}

async function startFaceTracking() {
    // Set the CSS dimensions of the canvas

    // Adjust the canvas resolution to account for device pixel ratio
    dpr = window.devicePixelRatio || 1; // Set the global dpr
    canvas.width = 640 * dpr;
    canvas.height = 480 * dpr;

    // Scale the canvas context to match the CSS size
    ctx.scale(dpr, dpr);

    // Disable anti-aliasing for crisp rendering
    ctx.imageSmoothingEnabled = false;

    video.width = 640;
    video.height = 480;

    // Hide the video element but keep it functional for MediaPipe
    video.style.display = "none";

    faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    faceMesh.onResults(processFaceMesh);

    async function detectFace() {
        await faceMesh.send({ image: video });
        requestAnimationFrame(detectFace);
    }

    detectFace();
}

async function processFaceMesh(results) {
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Optional: Set a background color (white to match your image)
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {
            const landmarkArray = [];

            // Loop through the first 468 landmarks and extract the x, y, z points
            for (let i = 0; i < Math.min(468, landmarks.length); i++) {
                const point = landmarks[i];
                const x = (point.x * canvas.width) / dpr; // Adjust for scaled context
                const y = (point.y * canvas.height) / dpr; // Adjust for scaled context
                landmarkArray.push([point.x, point.y, point.z]); // Normalized 3D points

                // Draw landmarks as circles (dots)
                ctx.beginPath();
                // Highlight 5 specific points per eye in red, others in blue
                if (i === 33 || i === 133 || i === 159 || i === 145 || // Left eye: 4 corners + 1 middle
                    i === 263 || i === 362 || i === 386 || i === 374) { // Right eye landmarks
                    ctx.fillStyle = "red";
                    ctx.arc(x, y, 1, 0, 2 * Math.PI); // Red dots (radius 1)
                } else {
                    ctx.fillStyle = "blue";
                    ctx.arc(x, y, 1, 0, 2 * Math.PI); // Blue dots (radius 1)
                }
                ctx.fill();
                
            }
            // Store the latest landmarks without calling backend
            latestLandmarks = landmarkArray;
        }
    }
}

// Send the latest landmarks only when the "Capture" button is clicked
captureBtn.addEventListener("click", () => {
    if (latestLandmarks) {
        sendLandmarksToBackend(latestLandmarks);
    } else {
        console.log("No landmarks detected yet.");
    }
});

async function sendLandmarksToBackend(landmarks) {
    const payload = {
        landmarks: landmarks  // Wrap the landmarks in an object
    };

    // Prepare payload data to save
    const payloadData = {
        firstName: currentUser.firstName,
        lastInitials: currentUser.lastInitials,
        stage: stages[currentStage].title,  // Current stage title
        timestamp: new Date().toISOString(),
        landmarks: landmarks
    };

    // Store payload data in localStorage
    let allPayloads = JSON.parse(localStorage.getItem("allPayloads")) || [];
    allPayloads.push(payloadData);
    localStorage.setItem("allPayloads", JSON.stringify(allPayloads));

    // Send request to backend
    const response = await fetch("https://emosnap.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();
    const stage = stages[currentStage];

    if (data.emotion && (data.emotion === stage.correctExpression || stage.correctExpression === 'neutral')) {
        currentUser.correctAnswers += 1;
        console.log("Correct answer! New correctAnswers:", currentUser.correctAnswers);
        renderAlert("🎉 Wow! You passed the stage!", "Next Stage");
        nextStage();
    } else {
        renderAlert("❌ Your face did not match.", "Try Again!");
    }

    console.log("After increment - currentUser.correctAnswers:", currentUser.correctAnswers); // Add this log
    localStorage.setItem("currentUser", JSON.stringify(currentUser));
}


// Function to reset the game and go to home screen
function resetGame() {
    saveProgress();
    currentStage = 0;  // Reset stage counter
    clearInterval(timerInterval);
    showResetConfirmation();
}

// Function to show the custom confirmation alert
function showResetConfirmation() {
    const overlay = document.createElement("div");
    overlay.classList.add("alert-overlay");

    overlay.innerHTML = `
        <div class="alert-box">
            <p>Are you sure you want to reset the game?</p>
            <div class="alert-buttons">
                <button class="alert-btn confirm" onclick="redirectToHomePage()">Yes, Reset</button>
                <button class="alert-btn cancel" onclick="closeAlert()">Cancel</button>
            </div>
        </div>
    `;

    document.body.appendChild(overlay);
}

// Function to close the alert
function closeAlert() {
    const overlay = document.querySelector(".alert-overlay");
    if (overlay) {
        document.body.removeChild(overlay);
    }
}

// Function to reset the game
function redirectToHomePage() {
    localStorage.removeItem("currentUser");
    window.location.href = "/";  // Redirect to index.html
}

function showEndGameConfirmationAlert() {
    const overlay = document.createElement("div");
    overlay.classList.add("alert-overlay");

    overlay.innerHTML = `
         <div class="alert-box">
            <p>Woohoo! You have completed all stages! 🎉</p>
            <div>
                <button class="alert-btn confirm" onclick="redirectToMetricsPage()">See My Game Results!</button>
            </div>
        </div>
    `;

    document.body.appendChild(overlay);
}


function redirectToMetricsPage() {
    window.location.href = "/metrics";  // Redirect to metrics.html
}

function renderAlert(message, buttonText = null) {
    // Create alert container
    const alertBox = document.createElement('div');
    alertBox.classList.add('alert-overlay');

    // Set innerHTML with dynamic content
    alertBox.innerHTML = `
        <div class="alert-box">
                <p>${message}</p>
                <div>
                    <button class="alert-btn confirm">${buttonText}</button>
                </div>
        </div>
    `;

    if (buttonText) {
        const button = alertBox.querySelector('.alert-btn');
        button.onclick = () => alertBox.remove();
    }

    // // Close the modal on click outside the alert
    // alertBox.onclick = (e) => {
    //     if (e.target === alertBox) {
    //         alertBox.remove();
    //     }
    // };

    // Append to the body
    document.body.appendChild(alertBox);
}
