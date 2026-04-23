# AI / Data Science Learning Roadmap (My Journey So Far)

A structured hands-on roadmap focused on learning by building real systems, projects, and understanding concepts deeply through implementation.

---

# 📌 Learning Philosophy

- Learn by building practical systems
- Prefer implementation + intuition over memorization
- Strong emphasis on debugging, optimization, and real-world pipelines
- Gradually move from foundations → advanced AI → multimodal systems

---

# 🟢 Stage 0 — Python Foundations

Before AI and Data Science, I completed a dedicated Python learning repository focused on mastering programming fundamentals.

🔗 https://github.com/ShailPatel27/Python-Learning

Covered:

- Python syntax and logic building
- Variables, strings, lists, tuples, dictionaries, sets
- Loops, functions, recursion
- File handling
- Object-oriented programming
- Inheritance & advanced OOP
- Exception handling
- Functional programming (`lambda`, `map`, `filter`, `reduce`)
- Virtual environments
- Flask basics
- Mini games and automation projects
- Mega projects like Jarvis Assistant and Auto Reply Chatbot

This repository built the programming foundation for all later work in databases, machine learning, PyTorch, and AI systems.

---

# 🟢 Stage 1 — Practical Programming Projects

## Wallet Management App

Built a full Python project with:

- Signup / Login system
- Session handling
- Transaction history
- File persistence
- Modular code structure

Later upgraded with databases.

---

# 🟢 Stage 2 — Database Systems

Learned PostgreSQL and CRUD operations.

Implemented PostgreSQL into Wallet Management system.

Topics covered:

- Connections
- Create table
- Insert
- Read
- Update
- Delete
- SQL schema design
- Python + PostgreSQL integration

---

# 🟢 Stage 3 — Data Science Foundations

---

## NumPy

Covered through modular notebook structure:

### M1 Basics
- Arrays
- Array creation
- Lists vs arrays
- Array functions

### M2 Indexing & Slicing
- 1D indexing
- 2D indexing
- Boolean masking

### M3 Array Operations
- Arithmetic
- Comparisons
- Aggregations

### M4 Matrix Operations
- Matrix multiplication
- Matrix functions
- Matrix properties

### M5 Reshaping
- reshape
- flatten
- transpose
- concatenate
- stack
- split

### M6 Random
- rand
- randint
- choice
- randn
- shuffle
- seed

---

## Pandas

Topics learned:

- DataFrames & Series
- Reading / writing CSV, Excel, JSON
- Filtering
- Indexing
- Cleaning dirty data
- Transformation
- GroupBy
- Aggregation
- Merge / Join / Concat
- DateTime handling

---

## Matplotlib

Topics learned:

- Plot basics
- Customization
- Common charts
- Subplots
- Object-oriented plotting
- Styling

---

## Seaborn

Topics learned:

- Distribution plots
- Categorical plots
- Relational plots
- Multivariate plots
- Themes / styling

---

# 🟢 Stage 4 — Machine Learning (Scikit-Learn)

Topics covered:

## Core Workflow

- Train/test split
- fit / predict
- Pipelines
- Feature scaling

## Supervised Learning

- Linear Regression
- Logistic Regression
- KNN
- Decision Trees
- Random Forests
- SVM

## Evaluation

- Accuracy
- Precision / Recall
- F1
- Confusion matrix
- Cross validation

## Unsupervised Learning

- KMeans
- Clustering basics

## Projects

- Mini projects:
    - Customer Segmentation
    - Customer Sex Prediction


---

# 🔵 Stage 5 — Deep Learning with PyTorch

---

# M1 Core PyTorch

Topics:

- Tensors
- Autograd
- Neural Networks
- Loss Functions
- Optimizers
- Training loops
- Visualization
- Validation split
- Dataset / Dataloader / batching
- Tensor shapes
- Save / Load models

---

# M2 Deep Learning

Topics:

- Feature scaling in neural nets
- Deeper networks
- Training stability
- Neural nets vs classical ML
- When not to use neural networks

Projects included.

---

# M3 CNNs

Topics:

- What CNNs are
- Loading image datasets
- Building CNNs
- CNN vs Linear models
- Error analysis
- Stronger CNN architectures
- CIFAR-10 training
- Performance improvements
- Data augmentation
- Transfer learning
- Using pretrained CNNs
- Fine-tuning

Mini Project:

- OCR (Optical Character Recognition)

---

# M4 Advanced Vision

Topics:

- Why deep CNNs fail without residuals
- Residual networks
- Skip connections
- Gradient flow
- Receptive field
- Width vs depth
- Representation learning
- Triplet loss
- Hard negative mining
- ArcFace / angular margin loss
- Vision Transformers
- Failure modes / bias

---

# M5 Face Recognition Systems

One of the biggest practical phases completed.

---

## Topics Covered

### Embeddings

- What is a face embedding
- Face → vector representation

### Similarity

- Cosine similarity
- Euclidean distance
- Threshold selection

### Recognition Systems

- Face database creation
- Matching unknown faces
- Multi-image per person averaging

### Real-Time Systems

- Camera operations
- Realtime recognition pipeline
- Multi-face recognition

### Optimization

- Frame skipping
- FPS smoothing
- Downscaled inference
- Idle mode

### Security

- Liveness detection
- Blink detection (EAR)
- Head movement detection

### Authentication

Built a full secure authentication system:

```text
Recognize Person
↓
Blink
↓
Turn Left
↓
Turn Right
↓
Authorized
````

### Sessions

* Multi-user timed sessions
* Remaining timer shown above faces

### Emotion Detection

* Real-time emotion detection intro

---

## Mini Project

### Secure Face Authentication System

An advanced real-time AI security pipeline.

---

# 🟣 M6 Video & Motion Recognition (Current / Next Major Phase)

This phase expands static vision → temporal intelligence.

---

## Planned Modules

### Object Detection

* What is object detection
* Bounding boxes
* IoU
* YOLO first detection
* Realtime object detection

### Tracking

* Object tracking
* DeepSORT
* Persistent IDs

### Motion

* Optical flow
* Motion detection

### Temporal AI

* Action recognition

### Mini Projects

* Hand Tracking Navigation System
* Face Tracking Navigation System
* Eye Tracking Navigation System
* Mediapipe Drawing Board

---

# 🟣 M7 Audio Models

## Fundamentals

* Audio as waveform
* Sampling rate
* Amplitude

## Signal Processing

* Fourier Transform
* Frequency domain

## Representations

* Spectrograms
* Mel spectrograms
* MFCC

## Deep Learning

* CNN for audio classification
* Keyword spotting
* Speech recognition basics

## Advanced

* Whisper architecture
* Speaker identification
* Emotion in voice

---

# 🟣 M8 Transformers & LLMs (Started)

Planned as one of the deepest phases.

---

## Roadmap

### Foundations

* Words & embeddings
* Tokenization
* BPE

### Attention

* Attention mechanism
* Query / Key / Value
* Multi-head attention

### Transformer Core

* Positional encoding
* Encoder / Decoder
* Full transformer architecture

### Language Modeling

* Next token prediction
* Perplexity

### Build Models

* Tiny GPT from scratch
* Tiny BERT from scratch

### Pretrained Models

* HuggingFace
* Inference
* Fine-tuning

### Scaling

* Schedules
* Gradient clipping
* Mixed precision

### Alignment

* RLHF
* Reward models

### Retrieval

* RAG
* Vector databases

### Multimodal

* Vision + language systems

### Mini Projects

* Domain chatbot
* Semantic search engine
* Custom LLM tools

---

# 🔴 Future Roadmap

---

## M9 Reinforcement Learning

Planned:

* MDPs
* Q-learning
* Policy gradients
* Actor-Critic
* PPO
* Grid-world agent
* Vision-based RL

---

## M10 Multimodal AI

Planned:

* Vision + Text
* Audio + Text
* Cross-modal embeddings
* CLIP-style systems
* Retrieval systems

---

# 🚀 Main Long-Term Project

# AEGIS

An evolving AI-powered security / monitoring system.

Started as:

* Object detection surveillance

Expanded toward:

* Face recognition
* Identity authentication
* Liveness verification
* Tracking
* Intelligent monitoring
* YOLO surveillance
* Motion alerts
* Emergency Identification

Future potential:

* IOT Support
* Audio detection
* Multimodal security AI

---

# 💡 Skills Built So Far

## Programming

* Python
* OOP
* Debugging
* Modular architecture

## Data

* NumPy
* Pandas
* Visualization

## ML

* Classical ML pipelines

## Deep Learning

* PyTorch
* CNNs
* Transfer learning
* Representation learning

## Computer Vision

* Face recognition
* Realtime detection
* Authentication systems
* Video pipelines

## Systems Engineering

* Optimization
* Real-time pipelines
* Multi-user logic
* Session systems

---


# 🧠 Learning Philosophy

> Build deeply. Understand fully. Scale gradually.