const fileInput = document.querySelector("#file-input");
const chooseButton = document.querySelector("#choose-button");
const analyzeButton = document.querySelector("#analyze-button");
const dropzone = document.querySelector("#dropzone");
const preview = document.querySelector("#preview");
const fileReadout = document.querySelector("#file-readout");
const statusStrip = document.querySelector(".status-strip");
const statusText = document.querySelector("#status-text");
const resultHero = document.querySelector("#result-hero");
const resultLabel = document.querySelector("#result-label");
const resultConfidence = document.querySelector("#result-confidence");
const meterFill = document.querySelector("#meter-fill");
const healthyProb = document.querySelector("#healthy-prob");
const unhealthyProb = document.querySelector("#unhealthy-prob");

let selectedFile = null;

function formatPercent(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

function setStatus(kind, text) {
  statusStrip.classList.toggle("loading", kind === "loading");
  statusStrip.classList.toggle("error", kind === "error");
  statusText.textContent = text;
}

function setResultState({ label = "Waiting", confidence = null, probabilities = null } = {}) {
  const normalized = label.toLowerCase();
  resultHero.classList.toggle("healthy", normalized === "healthy");
  resultHero.classList.toggle("unhealthy", normalized === "unhealthy");
  meterFill.classList.toggle("unhealthy", normalized === "unhealthy");

  resultLabel.textContent = label;
  resultConfidence.textContent = confidence === null ? "--" : `${formatPercent(confidence)} confidence`;
  meterFill.style.width = confidence === null ? "0%" : `${Math.max(2, confidence * 100)}%`;

  healthyProb.textContent = probabilities?.healthy === undefined ? "--" : formatPercent(probabilities.healthy);
  unhealthyProb.textContent =
    probabilities?.unhealthy === undefined ? "--" : formatPercent(probabilities.unhealthy);
}

function pickFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("error", "Select an image file");
    return;
  }

  selectedFile = file;
  const objectUrl = URL.createObjectURL(file);
  preview.src = objectUrl;
  dropzone.classList.add("has-image");
  fileReadout.textContent = `${file.name} · ${Math.round(file.size / 1024)} KB`;
  analyzeButton.disabled = false;
  setStatus("ready", "Ready");
  setResultState();
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read the image"));
    reader.readAsDataURL(file);
  });
}

async function analyzeImage() {
  if (!selectedFile) return;

  analyzeButton.disabled = true;
  setStatus("loading", "Analyzing");

  try {
    const imageData = await fileToDataUrl(selectedFile);
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: selectedFile.name,
        image_data: imageData,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed");
    }

    setResultState(payload);
    setStatus("ready", "Complete");
  } catch (error) {
    setStatus("error", error.message);
    setResultState({ label: "Error" });
  } finally {
    analyzeButton.disabled = false;
  }
}

chooseButton.addEventListener("click", () => fileInput.click());
analyzeButton.addEventListener("click", analyzeImage);
fileInput.addEventListener("change", (event) => pickFile(event.target.files[0]));

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag-over");
  pickFile(event.dataTransfer.files[0]);
});
