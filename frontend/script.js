// Configuration - Change this to your FastAPI backend URL
const API_BASE_URL = "http://127.0.0.1:8000"

// Wait for DOM to be fully loaded before accessing elements
document.addEventListener("DOMContentLoaded", () => {
  console.log("[AutoML] DOM loaded, initializing...")

  // DOM Elements
  const uploadArea = document.getElementById("uploadArea")
  const csvFileInput = document.getElementById("csvFile")
  const fileInfo = document.getElementById("fileInfo")
  const fileName = document.getElementById("fileName")
  const removeFileBtn = document.getElementById("removeFile")
  const columnPreview = document.getElementById("columnPreview")
  const columnList = document.getElementById("columnList")
  const targetColumnInput = document.getElementById("targetColumn")
  const trainForm = document.getElementById("trainForm")
  const trainBtn = document.getElementById("trainBtn")
  const trainResult = document.getElementById("trainResult")
  const trainError = document.getElementById("trainError")
  const trainErrorText = document.getElementById("trainErrorText")

  // Result display elements
  const resultProblemType = document.getElementById("resultProblemType")
  const resultBestModel = document.getElementById("resultBestModel")
  const resultScore = document.getElementById("resultScore")
  const resultHyperparams = document.getElementById("resultHyperparams")

  const predictForm = document.getElementById("predictForm")
  const predictDataInput = document.getElementById("predictData")
  const predictBtn = document.getElementById("predictBtn")
  const predictResult = document.getElementById("predictResult")
  const predictionValue = document.getElementById("predictionValue")
  const predictError = document.getElementById("predictError")
  const predictErrorText = document.getElementById("predictErrorText")

  console.log("[AutoML] Result elements check:")
  console.log("[AutoML] trainResult:", trainResult ? "FOUND" : "MISSING")
  console.log("[AutoML] resultProblemType:", resultProblemType ? "FOUND" : "MISSING")
  console.log("[AutoML] resultBestModel:", resultBestModel ? "FOUND" : "MISSING")
  console.log("[AutoML] resultScore:", resultScore ? "FOUND" : "MISSING")
  console.log("[AutoML] resultHyperparams:", resultHyperparams ? "FOUND" : "MISSING")

  if (!uploadArea || !trainForm || !predictForm) {
    console.error("[AutoML] Required DOM elements not found!")
    return
  }

  let selectedFile = null
  let detectedColumns = []

  // File Upload Handlers
  uploadArea.addEventListener("click", () => {
    csvFileInput.click()
  })

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault()
    uploadArea.classList.add("dragover")
  })

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover")
  })

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault()
    uploadArea.classList.remove("dragover")
    const files = e.dataTransfer.files
    if (files.length > 0 && files[0].name.endsWith(".csv")) {
      handleFileSelect(files[0])
    }
  })

  csvFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelect(e.target.files[0])
    }
  })

  removeFileBtn.addEventListener("click", () => {
    selectedFile = null
    detectedColumns = []
    csvFileInput.value = ""
    fileInfo.classList.add("hidden")
    columnPreview.classList.add("hidden")
    uploadArea.classList.remove("hidden")
    targetColumnInput.value = ""
  })

  function handleFileSelect(file) {
    selectedFile = file
    fileName.textContent = file.name
    uploadArea.classList.add("hidden")
    fileInfo.classList.remove("hidden")

    // Read first line to get column names
    const reader = new FileReader()
    reader.onload = (e) => {
      const text = e.target.result
      const firstLine = text.split("\n")[0]
      const delimiter = firstLine.includes(";") ? ";" : ","
      detectedColumns = firstLine.split(delimiter).map((col) => col.trim().replace(/"/g, ""))

      if (detectedColumns.length > 0) {
        columnList.innerHTML = ""
        detectedColumns.forEach((col) => {
          if (col) {
            const tag = document.createElement("span")
            tag.className = "column-tag"
            tag.textContent = col
            tag.addEventListener("click", () => {
              document.querySelectorAll(".column-tag").forEach((t) => t.classList.remove("selected"))
              tag.classList.add("selected")
              targetColumnInput.value = col
            })
            columnList.appendChild(tag)
          }
        })
        columnPreview.classList.remove("hidden")
      }
    }
    reader.readAsText(file)
  }

  // Train Form Handler
  trainForm.addEventListener("submit", async (e) => {
    e.preventDefault()
    console.log("[AutoML] Train form submitted")

    if (!selectedFile) {
      showTrainError("Please select a CSV file first.")
      return
    }

    const targetColumn = targetColumnInput.value.trim()
    if (!targetColumn) {
      showTrainError("Please enter or select a target column name.")
      return
    }

    if (detectedColumns.length > 0 && !detectedColumns.includes(targetColumn)) {
      showTrainError(`Column "${targetColumn}" not found in CSV. Available columns: ${detectedColumns.join(", ")}`)
      return
    }

    setTrainLoading(true)
    hideTrainMessages()

    console.log("[AutoML] Starting fetch to:", `${API_BASE_URL}/train`)
    console.log("[AutoML] File:", selectedFile.name)
    console.log("[AutoML] Target column:", targetColumn)

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)
      formData.append("target_column", targetColumn)

      console.log("[AutoML] Sending request...")
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: "POST",
        body: formData,
      })

      console.log("[AutoML] Response received, status:", response.status)
      console.log("[AutoML] Response ok:", response.ok)

      const data = await response.json()
      console.log("[AutoML] Parsed JSON data:", JSON.stringify(data, null, 2))

      if (!response.ok) {
        console.log("[AutoML] Response not OK, throwing error")
        throw new Error(data.detail || "Training failed. Check if the target column exists in your CSV.")
      }

      console.log("[AutoML] Calling showTrainSuccess...")
      showTrainSuccess(data)
      console.log("[AutoML] showTrainSuccess completed")
    } catch (error) {
      console.error("[AutoML] Caught error:", error)
      console.error("[AutoML] Error name:", error.name)
      console.error("[AutoML] Error message:", error.message)
      showTrainError(
        error.message || "Failed to connect to the server. Make sure the backend is running on " + API_BASE_URL,
      )
    } finally {
      setTrainLoading(false)
      console.log("[AutoML] Train handler finished")
    }
  })

  // Predict Form Handler
  predictForm.addEventListener("submit", async (e) => {
    e.preventDefault()

    const inputData = predictDataInput.value.trim()

    if (!inputData) {
      showPredictError("Please enter input data.")
      return
    }

    let parsedData
    try {
      parsedData = JSON.parse(inputData)

      if (Array.isArray(parsedData)) {
        showPredictError('Please provide a single JSON object, not an array. Example: {"feature1": 10}')
        return
      }

      if (typeof parsedData !== "object" || parsedData === null) {
        showPredictError("Please provide a valid JSON object.")
        return
      }
    } catch (err) {
      showPredictError("Invalid JSON format. Please check your input. Make sure to use double quotes for keys.")
      return
    }

    setPredictLoading(true)
    hidePredictMessages()

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(parsedData),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed. Make sure you have trained a model first.")
      }

      showPredictSuccess(data)
    } catch (error) {
      console.error("[AutoML] Prediction error:", error)
      showPredictError(error.message || "Failed to connect to the server. Make sure the backend is running.")
    } finally {
      setPredictLoading(false)
    }
  })

  // Helper Functions
  function setTrainLoading(loading) {
    const btnText = trainBtn.querySelector(".btn-text")
    const btnLoader = trainBtn.querySelector(".btn-loader")
    trainBtn.disabled = loading
    btnText.classList.toggle("hidden", loading)
    btnLoader.classList.toggle("hidden", !loading)
  }

  function setPredictLoading(loading) {
    const btnText = predictBtn.querySelector(".btn-text")
    const btnLoader = predictBtn.querySelector(".btn-loader")
    predictBtn.disabled = loading
    btnText.classList.toggle("hidden", loading)
    btnLoader.classList.toggle("hidden", !loading)
  }

  function hideTrainMessages() {
    trainResult.classList.add("hidden")
    trainError.classList.add("hidden")
  }

  function hidePredictMessages() {
    predictResult.classList.add("hidden")
    predictError.classList.add("hidden")
  }

  function showTrainSuccess(data) {
    console.log("[AutoML] showTrainSuccess called with:", JSON.stringify(data, null, 2))

    // Validate data
    if (!data) {
      console.error("[AutoML] No data received")
      showTrainError("No response data received from backend.")
      return
    }

    // Extract details - handle both {details: {...}} and direct object
    const details = data.details || data
    console.log("[AutoML] Extracted details:", JSON.stringify(details, null, 2))

    // Check if we have valid details
    if (!details || typeof details !== "object") {
      console.error("[AutoML] Invalid details object")
      showTrainError("Invalid response format from backend.")
      return
    }

    try {
      // Problem type
      if (resultProblemType) {
        const problemType = details.problem_type || "Unknown"
        resultProblemType.textContent = problemType.toString().toUpperCase()
        console.log("[AutoML] Set problem_type:", problemType)
      } else {
        console.error("[AutoML] resultProblemType element not found!")
      }

      // Best model - THIS IS THE KEY FIX
      if (resultBestModel) {
        const bestModel = details.best_model || "N/A"
        resultBestModel.textContent = bestModel.toString()
        console.log("[AutoML] Set best_model:", bestModel)
      } else {
        console.error("[AutoML] resultBestModel element not found!")
      }

      // Score
      if (resultScore) {
        let scoreText = "N/A"
        if (details.problem_type === "regression") {
          if (details.test_score != null && !isNaN(details.test_score)) {
            scoreText = (details.test_score * 100).toFixed(2) + "% (R²)"
          } else if (details.best_score != null && !isNaN(details.best_score)) {
            scoreText = Math.abs(details.best_score).toFixed(4) + " (MSE)"
          }
        } else {
          if (details.best_score != null && !isNaN(details.best_score)) {
            scoreText = (details.best_score * 100).toFixed(2) + "%"
          }
        }
        resultScore.textContent = scoreText
        console.log("[AutoML] Set score:", scoreText)
      } else {
        console.error("[AutoML] resultScore element not found!")
      }

      // Hyperparameters
      if (resultHyperparams) {
        if (details.best_hyperparameters && Object.keys(details.best_hyperparameters).length > 0) {
          const cleaned = {}
          for (const [k, v] of Object.entries(details.best_hyperparameters)) {
            cleaned[k.replace("model__", "").replace("poly__", "")] = v
          }
          resultHyperparams.textContent = JSON.stringify(cleaned, null, 2)
        } else {
          resultHyperparams.textContent = "Default parameters used"
        }
        console.log("[AutoML] Set hyperparameters")
      } else {
        console.error("[AutoML] resultHyperparams element not found!")
      }

      if (trainResult) {
        trainResult.removeAttribute("style")
        trainResult.classList.remove("hidden")
        console.log("[AutoML] trainResult shown - classList:", trainResult.classList.toString())
      } else {
        console.error("[AutoML] trainResult element not found!")
      }

      // Hide any previous errors
      if (trainError) {
        trainError.classList.add("hidden")
      }

      console.log("[AutoML] Train result displayed successfully")
    } catch (err) {
      console.error("[AutoML] Error in showTrainSuccess:", err)
      showTrainError("Error displaying results: " + err.message)
    }
  }

  function showTrainError(message) {
    trainErrorText.textContent = message
    trainError.classList.remove("hidden")
    trainResult.removeAttribute("style")
    trainResult.classList.add("hidden")
  }

  function showPredictSuccess(data) {
    const prediction = data.prediction
    if (Array.isArray(prediction)) {
      predictionValue.textContent = prediction.join(", ")
    } else {
      predictionValue.textContent = prediction
    }
    predictResult.classList.remove("hidden")
  }

  function showPredictError(message) {
    predictErrorText.textContent = message
    predictError.classList.remove("hidden")
  }

  // Smooth scroll for navigation
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault()
      const target = document.querySelector(this.getAttribute("href"))
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        })
      }
    })
  })
})
