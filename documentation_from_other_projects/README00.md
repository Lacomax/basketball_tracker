# 🏀 AI-Powered Basketball Game Analyzer (v1)

> **Personalized & Complete Version of an Academic Project**  
> 🔁 Rewritten, restructured, and integrated from scratch by Hana FEKI

---

## 🎥 Demo Video

![Demo of AI-Powered Basketball Game Analyzer](output_videos/Video_1_output.gif)


---

## 📘 Project Context

This project was initially part of a **group academic project at ENSTA Paris** during the 2024–2025 academic year. We were a team of **10 students**, split into **5 subgroups**, each responsible for a specific part of the system (detection, tracking, analytics, etc.).

Due to tight academic deadlines and the distributed nature of the work, the project:
- Was not fully completed
- Had parts that remained unintegrated
- Lacked a unified and polished implementation

---

## ❤️ Why This Version Exists (v1)

As I was **deeply passionate** about the topic, I decided to **rebuild the entire system from A to Z** by myself:
- Rewriting every component
- Organizing everything into a single, consistent pipeline
- Fixing bugs and improving the original implementation
- Adding missing features and enhancements

> 📁 The original version (v0) is available here: [Version 0 - GitHub Repo](https://github.com/HanaFEKI/AI_BasketBall_Analysis_v0)

> ✅ This repository is the **personalized, cleaned-up, and extended version (v1)**.

---

## ⚙️ How It Works

This system analyzes basketball games from video using computer vision and AI:

1. **🎯 Object Detection (YOLO)**  
   Detect players and the basketball in each frame.

2. **🧭 Object Tracking (ByteTrack)**  
   Track players and the ball across video frames.

3. **🎨 Team Classification (Zero-Shot with Hugging Face)**  
   Automatically assign players to teams based on jersey colors using a zero-shot image classifier powered by [Fashion CLIP](https://huggingface.co/patrickjohncyh/fashion-clip).

5. **📍 Court Keypoint Detection**  
   Detect basketball court landmarks using a keypoint detection model trained on a labeled dataset.

6. **🔄 Perspective Transformation**  
   Convert broadcast view into a **top-down tactical map** using homography and real-world court dimensions.

7. **📊 Analytics**  
   - Count **passes** and **interceptions**  
   - Compute **ball possession** percentage  
   - **Speed and Distance Calculation**

> ⚠️ To keep the video output uncluttered and visually clear, **speed and distance metrics are not drawn directly on the video frames**. Instead, they are computed separately and saved for further analysis or plotting.

---

## 🧗‍♀️ Challenges Faced

- **Video Quality:** Variability in broadcast footage resolution and lighting made detection and tracking difficult.  
- **Homography Estimation:** Accurate court perspective transformation required fine-tuning and was time-consuming.  
- **Ball Handler Identification:** Distinguishing the player with ball possession posed significant complexity.  
- **YOLO Model Selection:** Choosing and training the best object detection models took extensive experimentation.

---

## 🚀 Future Work & Perspectives

- Implement **OCR for jersey number recognition** to reliably identify players.  
- Improve **player re-identification** across multiple cameras or games.  
- Enhance **event detection**, including fouls, shots, and rebounds.

---
