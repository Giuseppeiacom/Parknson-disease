Per un corretto utilizzo utilizzare python 3.10.

This project implements a complete pipeline for analyzing facial movements from video data in order to distinguish between healthy subjects and patients with Parkinson’s disease, as well as to study emotional expressions.

## 📌 Feature Extraction Pipeline:

This module is responsible for processing all .mov videos contained in the root directory and extracting meaningful facial features for subsequent analysis.

The pipeline begins by scanning the root directory to identify all video files with .mov extension. For each video, the system assigns a label based on the patient’s condition: 0 for healthy patients (HP) and 1 for patients with Parkinson’s disease (SP). Additionally, each video is associated with a corresponding emotion label.

Each video is then passed to the function extract_features_for_video, which processes it frame by frame. For every frame, the MediaPipe framework is used to detect 468 facial landmarks, covering key regions such as the eyes, mouth, eyebrows, and cheeks. To ensure consistency across different subjects and video conditions, these landmarks are normalized with respect to the size of the face, using the distance between the nose (landmark 1) and the chin (landmark 152). All extracted data from the frames are stored in a list called all_frames_data.

To establish a reference baseline, the algorithm analyzes the first 100 frames of the video (approximately 3.5 seconds). Within this segment, a sliding window of 30 frames (corresponding to about 1 second at 30 FPS) is applied. This window moves sequentially across the frames (e.g., frames 0–30, 1–31, 2–32, and so on), allowing the system to evaluate different temporal segments.

For each window position, the variance of the facial landmarks is computed. High variance indicates significant movement (such as speaking or expressive behavior), while low variance corresponds to minimal movement, representing a neutral or stable state. The window with the lowest variance is selected as the baseline. From this most stable segment, statistical measures such as the mean and standard deviation are calculated.

The reaction time is then determined by identifying the first frame in which the movement exceeds a predefined threshold. The time elapsed from the beginning of the video to this frame represents the subject’s reaction time.

All extracted information is organized into a structured dictionary called samples, which includes the computed features, reaction time, patient label, and emotion label. Finally, these data are aggregated into a dataset with entries identified by names such as affabile_mar_..., which can be used for machine learning tasks or statistical analysis.

## 📌 Validation, Stabilization and Aggregation

To verify the correctness of facial tracking, the function visualizza_tutto_unificato is used as a validation tool. This allows a visual inspection of the detected landmarks over time, ensuring that the system consistently follows the subject’s face.

To make the analysis robust to head movements, a stabilization strategy is applied through the function get_normalized_displacement_smart. In this approach, the nose tip (landmark 1) is used as a reference point. All other landmarks are computed relative to it. This ensures that rigid head movements do not affect the analysis: if a subject moves their head but keeps the same facial expression, the detected movement remains effectively zero.

A temporal strategy is also applied by comparing expressive frames with the neutral baseline. This allows the computation of a displacement vector, representing how each landmark moves from a neutral to an emotional state.

#### 📌 Classification Pipeline

This module is responsible for classifying subjects as **healthy** or affected by **Parkinson’s disease** using the features extracted from the video analysis phase.

The process begins by reading the video filenames in order to identify each patient. Labels are assigned automatically based on naming conventions: if the filename contains `hp`, the subject is labeled as **0 (healthy)**; if it contains `sp`, the subject is labeled as **1 (Parkinson)**.

The original data consist of frame-by-frame features. To make them suitable for classification, these temporal data are summarized into three key statistical descriptors:

* **Mean**, representing the average facial movement
* **Standard deviation**, capturing the variability of the expression over time
* **Maximum value**, indicating the peak intensity of facial movement

Since the goal is to perform classification at the patient level, the dataset is then **flattened** by aggregating all samples belonging to the same subject. This is done by iterating over all patient IDs (e.g., using a loop such as `for pid in patient_ids:`), resulting in a single feature vector per patient.

Given the limited number of samples and the associated risk of overfitting, a feature selection step is applied using the **Boruta** algorithm. This method identifies only those features that have real predictive power by comparing them with randomized “shadow features,” discarding irrelevant or noisy variables in a statistically robust manner.

Several classification models are then evaluated, including Random Forest, Support Vector Machine (SVM), and other standard algorithms. Due to the small dataset size (only 19 patients), model evaluation is performed using **Leave-One-Out Cross-Validation (LOOCV)**. This approach ensures that each sample is used once as a test instance while the remaining data are used for training, providing a reliable and unbiased evaluation without wasting data.

Among the tested models, **Random Forest** achieves the best performance. The final evaluation focuses on clinically relevant metrics:

* **Sensitivity**, which measures the ability to correctly identify patients with Parkinson’s disease
* **Specificity**, which measures the ability to correctly identify healthy subjects

The results indicate that the model has high sensitivity, meaning it is effective at detecting pathological cases. This characteristic makes it particularly suitable as a **screening tool**, where the priority is to minimize missed diagnoses rather than false positives.
