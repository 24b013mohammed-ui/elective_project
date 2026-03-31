Name : VK Mohammed Shifaz
Registeration Number : TCR24CS070

# Shifaz Forecaster Engine

A premium, full-stack quantitative analytics engine designed for processing sequential market data using deep Convolutional Neural Networks (CNN) and Short-Time Fourier Transform (STFT) signal processing.

This project bridges highly sophisticated qualitative deep learning paradigms with a visually clean, minimalist internal dashboard, empowering rapid model iteration and frequency transforms evaluation right from your browser.

---

## 🚀 System Initialization

The architecture is loosely coupled between a robust Python backend inference node and a sleek React.js (Vite) frontend interface. To utilize the Engine, invoke both sub-systems concurrently.

### 1. Start the Backend Inference Node (Python/FastAPI)
The backend pipeline handles tensor computations, STFT transformations, and live network convergence.
1. Open a terminal and navigate to the `backend` directory.
2. Ensure you have installed all computational dependencies: `pip install -r requirements.txt`.
3. Simply execute the start script:
   - **Windows:** Run `start_backend.bat` (this automatically provisions `python backend_api.py`).
4. Maintain the terminal process. The engine will idle on standby to accept initialization parameters from the visual interface.

### 2. Start the Diagnostic Interface (React/Vite)
The frontend serves as the primary visual command center, equipped with an intuitive Left-Sidebar navigation layout and specialized data analysis panels.
1. Open a new, separate terminal and navigate to the `frontend` directory.
2. Initialize the Node package cache (first-time execution only):
   ```bash
   npm install
   ```
3. Boot the Vite development server:
   ```bash
   npm run dev
   ```
4. Access the interface via the locally hosted URI (typically `http://localhost:3000` or `http://localhost:5173`) to launch the Shifaz Forecaster Engine.

---

## 🧮 Architecture & Tensor Operations

Our system evaluates sequential time-series patterns through a highly unconventional mathematical lens: **Signal Processing + Spatial Vision Analysis**.

### 1. Vector Transformation via STFT
Standard approaches feed raw 1D arrays directly to recurrent networks. Instead, we leverage the **Short-Time Fourier Transform (STFT)** to structurally mutate 1D sequences into 2D **Frequency Fields** (time-frequency spatial domains).

- **The Logic:** Sequential data exhibits hidden periodic micro-fluctuations and cyclic trends. By transforming this sequence into the frequency domain via overlapping host windows (e.g., 20-step intervals), we mathematically isolate these invisible frequencies.
- **The Equation:** $S(t,f) = \left| \sum_{n} x[n] w[n-t] e^{-j 2\pi f n / N} \right|^2$
- Each independent channel undergoes discrete STFT processing. The magnitude-squared tensor arrays form a **multi-dimensional matrix**, operating virtually identically to an RGB image.

### 2. Deep Convolutional Architecture 
Once the tensor is formatted, it propagates through our custom Convolutional Neural Network layer stack.

- **Extraction Mechanism:** `Conv2d` layers paired with non-linear `ReLU` activations map spatial structural boundaries within the frequency matrices. Sub-sampling (`MaxPool2d`) continuously reduces computational dimensionality.
- **Output Regression:** Flattened tensors traverse fully connected linear nodes. Since the Engine targets a continuous trajectory rather than classification, the final output neuron calculates a normalized regression vector predicting the sequence state at the specified T+5 chronological horizon.

---

## 📊 Left-Sidebar Interface Navigation

The frontend translates extreme computational complexity into a highly readable, clinical UI split across several specialized modules:

### 1. Prediction Command Center (Home)
- **Primary Function:** Provides an instantaneous, birds-eye diagnostic view of the Engine's current state. 
- **Analytical Output:** Aggregates primary execution metrics (Coefficient of Determination $R^2$, Absolute Deviation, Percentage Errors) and presents high-level routing logic in a visually distinct asymmetrical grid.

### 2. Engine Parameters (Config)
- **Primary Function:** A sophisticated dual-pane interface to define hyperparameter boundaries and structural constants without altering the codebase.
- **Analytical Output:** Controls asset identifiers, historical range lookbacks, and optimization targets (Batch Count, Iteration Limits, Pace).

### 3. Convergence Control (Training)
- **Primary Function:** A diagnostic monitoring layer tracking gradient descent progression.
- **Analytical Output:** Utilizes dynamic bounding Area Charts to visualize both Internal Error (Training Loss) and Validation Error iteratively, immediately exposing over-fitting tendencies.

### 4. Forecast Horizons (Predictions)
- **Primary Function:** The explicit post-execution analysis matrix comparing calculated tensor outputs versus true values.
- **Analytical Output:** Generates a dense, highly structured numerical table calculating the delta deviation per step, allowing analysts to manually verify directional integrity.

### 5. Frequency Field Visualizer (Spectrograms)
- **Primary Function:** Exposes the raw signal mechanics bridging the temporal and spatial domains.
- **Analytical Output:** Defines the methodological constants of our STFT operations (e.g., specific dimensions like 5 × 128 × 18) and clarifies the hidden multi-band processing logic used by the engine.

### 6. Evaluation Protocol & Diagnostics
- **Primary Function:** Deep quantitative assessment of residual spreads.
- **Analytical Output:** Generates clean, light-mode Variance Density Histograms and Residual Drift scatter plots. Maps exactly where the network generates systemic biases so the parameters can be continuously iterated upon.
