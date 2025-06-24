# 🧠 CUDA Neural Network from Scratch

A fully functional feedforward neural network built from **scratch** using **C++ and raw CUDA**, running in a **Visual Studio project**. This project demonstrates the inner workings of neural networks — without using any high-level frameworks like PyTorch, cuDNN, or cuBLAS.

---

## 💭 Table of Content
- [🚀 Features](#-features)
- [🧠 Model Architecture](#-model-architecture)
- [📸 Dataset Format](#-dataset-format)
- [📁 Project Structure](#-project-structure)
- [🧑‍💻 Development Environment](#-development-environment)
- [🧪 Example Training Output](#-example-training-output)
- [▶️ Running the Project](#-running-the-project)
- [📌 Goals of This Project](#-goals-of-this-project)
- [✅ What is Achieved](#-what-is-achieved)
- [🔮 Future Ideas (Optional)](#-future-ideas-optional)
- [📄 License](#-license)
- [🙋‍♂️ Author](#-author)

## 🚀 Features

- 🔢 Custom layers: `Linear`, `ReLU`, `Sigmoid` (with CUDA forward/backward kernels)
- 🧱 `Sequential` container to build networks dynamically
- 🎯 Binary Cross-Entropy (`BCELoss`) loss function
- ✅ Accuracy metric (`BinaryAccuracy`) with CUDA atomic operations
- 📦 Fully implemented training and validation loops
- 🗃️ CSV data loading with image normalization

---

## 🧠 Model Architecture

```

Input (64x64 grayscale = 4096)
↓
Linear (4096 → 1024)
↓
ReLU
↓
Linear (1024 → 1)
↓
Sigmoid

```

---

## 📸 Dataset Format

- Grayscale image data flattened into CSV rows (64×64 = 4096 features)
- Each row starts with the **label**: `0 = dog`, `1 = cat`
- All pixel values are normalized to the `[0, 1]` range (by dividing by 255)
- Example CSV row:
```

0, 123, 45, 67, ..., 255

```

---

## 📁 Project Structure

```

.
├── layers/         # Neural network layers (Linear, ReLU, Sigmoid, Sequential)
├── loss/           # Binary Cross-Entropy loss
├── accuracy/       # Binary accuracy implementation
├── utils/          # Normalization, matrix printing, helper functions
├── data/           # CSV data loading and data files
├── train/          # Training loop
├── validate/       # Validation logic
├── main.cu         # Entry point

```

---

## 🧑‍💻 Development Environment

- 🧠 **Language:** C++17
- ⚡ **GPU:** CUDA-enabled NVIDIA GPU (developed and tested on NVIDIA GTX 1070)
- 🧰 **IDE:** Visual Studio 2019 or newer
- 🛠️ **CUDA Toolkit:** Version 11.0 or higher

---

## 🧪 Example Training Output

```

Begin training...
[Epoch 1] Loss: 4.61324, Accuracy: 0.472727
[Epoch 51] Loss: 0.252393, Accuracy: 0.890909
...
[Epoch 100] Loss: 0.225266, Accuracy: 0.909091
Training complete!

Begin validating...
[Batch 1] Accuracy: 0.5
[Batch 3] Accuracy: 0.7
...
Average accuracy: 0.533333

````

---

## ▶️ Running the Project

1. Clone the project into a directory:
   ```bash
   git clone https://github.com/Matan-Vinkler/neural-network-cuda.git
   ```

2. Open the `.sln` file in **Visual Studio**.

3. Ensure your CUDA Toolkit is installed and the correct `Compute Capability` is selected in project settings (e.g., `sm_61`, `sm_75`, etc.).

4. Set `main.cu` as the startup file.

5. Prepare your datasets:

   * `data/train_data.csv`
   * `data/val_data.csv`

6. Hit **Run (F5)**.

---

## 📌 Goals of This Project

* ✅ Demonstrate **low-level CUDA programming**
* ✅ Learn and implement the **mechanics of neural networks**
* ✅ Build a **modular, testable neural net library**
* ✅ Train a model on real grayscale image data (cats vs. dogs)

This is not just an ML project — it’s a systems-level engineering exercise in GPU computing, memory management, and performance-aware design.

---

## ✅ What is Achieved

* Built a neural network with layers, activations, and loss from scratch
* Implemented forward/backward passes using raw CUDA kernels
* Created a full training loop with accuracy evaluation
* Clean C++/CUDA project with modular design

---

## 🔮 Future Ideas (Optional)

* Add Dropout or BatchNorm layers
* Add CNN layers and image processing methods
* Implement other loss functions (e.g., MSE, CrossEntropy for multiclass)
* Add support for RGB images
* Write a Python inference wrapper using `pybind11`
* Log metrics to CSV for visualization with Python

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Matan** — [GitHub](https://github.com/Matan-Vinkler) | [Linkedin](https://www.linkedin.com/in/matan-vinkler-673120201)

---