# Real Time Sign Language Translator Using Computer Vision

## MSML640 Computer Vision
### Final Project Report
### Members: Likhon, Waseem, Geona, Hans, Yajat
### Repository Link: https://github.com/likhongomes/Real-Time-Sign-Language-Translator.git 

## How to run the code
1. Clone the repository
```
git clone git@github.com:likhongomes/Real-Time-Sign-Language-Translator.git
cd Real-Time-Sign-Language-Translator
```
2. Create a virtual environment and install dependencies
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. Download the MS-ASL dataset by running the following command:
```python preprocess/downoad.py```
4. Preprocess the dataset to extract hand landmarks:
```python utils.py prepocess```
5. Train the model using the following command:
```python train.py```
6. Evaluate the model using the following command:
```python utils.py evaluate```
7. Inference using webcam:
```python inference.py```

## Introduction
Despite rapid progress in speech recognition and text based translation, communication between Deaf or Hard of Hearing (DHH) signers and nonsigners still relies heavily on human interpreters. Automatic American Sign Language (ASL) translation is particularly challenging because meaning is encoded not only in discrete handshapes, but also in motion, timing and in full ASL, facial expressions and body posture. Unlike spoken language, where audio can be modeled as a one dimensional signal, ASL requires spatiotemporal knowledge of 2D video and 3D body dynamics. These challenges make it difficult to implement practical ASL translation tools in everyday environments. 

In this project, we explore a vision based pipeline for recognizing ASL signs from video with the goal of moving toward realtime translation. Rather than operating directly on raw pixels, our system first downloads and trims sign videos from the MS-ASL dataset using provided YouTube URLs and timestamped annotations and organizes them into respective class folders for training. We then extract 3D hand keypoints from each frame using a MediaPipe based hand landmark detector, which outputs 21 landmarks per hand and caches these sequences for efficient reuse. This landmark representation compresses each video into a sequence of structured skeletal features that emphasize hand shape and motion while reducing background noise and lighting variation. 

On top of these landmark sequences, we train several temporal neural network models including bidirectional LSTMs with attention, Transformer encoders, and Temporal Convolutional Networks (TCNs) to classify isolated signs from short video clips. The training pipeline includes data augmentation, mixed precision optimization, learning rate scheduling, and checkpointing, which allows us to compare architectures under consistent experimental settings. For evaluation, we compute top-1 and top-5 accuracy and generate confusion matrices and per class metrics, along with qualitative visualizations of model predictions on sample clips. Finally, we prototype a real time inference module that connects a webcam to the trained model, performs live hand landmark extraction, and applies sliding window, temporally smoothed predictions to display the most likely sign on the video stream.

While our original proposal envisioned a full realtime ASL fingerspelling translator, in this work we focus on building and analyzing an end to end sign recognition system that operates on isolated signs in a controlled setting and we treat the live demo as a proof of concept rather than a fully polished product. 

## Related Work
Research on sign-language recognition has evolved tremendously over the years, driven by advances in computer vision and deep learning. Earlier breakthroughs came from curated datasets such as MS-ASL (Joze et al., 2018) and WLASL (Li et al., 2019), which provided thousands of labeled sign videos recorded under varied conditions. These datasets enabled researchers and scientists alike to train Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and I3D-based models capable of learning spatiotemporal patterns in sign language. Survey work such as Zhang et al. (2024) and Subburaj et al. (2022) further contextualized the field’s progression, highlighting the challenges of signer variability, limited training data, and immense computational requirements inherent to RGB-based recognition

As the field matured, researchers shifted from raw video input to keypoint-based representations, largely enabled by frameworks such as MediaPipe and OpenPose. Studies such as John and Sherif (2022) demonstrated that training classifiers directly on extracted hand landmarks significantly reduced the noise sensitivity and computational cost compared to image-based models. Academic works including Alsharif et al. (2025) showed that landmark-driven pipelines are highly effective for real time Sign Language Recognition (SLR), providing low-latency performance suitable for user-facing applications. These contributions have marked a major step towards accessible SLR systems by reducing hardware requirements while maintaining strong recognition accuracy. 

More recent directions have started to explore lightweight architectures and model compression to push SLR closer to deployable real-world systems. The KD-MSLRT model (Li et al., 2025) introduced a MediaPipe-based architecture combined with knowledge distillation and later evaluated INT8 quantization, demonstrating minimal accuracy loss when compressing models for faster inference. Complementary research on embedded SLR systems, such as Sandhya et al. (2023), confirmed that quantization drastically accelerates inference on low-power hardware. Meanwhile, transformer-based systems like BdSLW401 (Rubadiyeat et al., 2025) showed the potential of more expressive sequence models for sign recognition, though these models typically rely on high-quality curated datasets and still remain computationally expensive.

Although these works have pushed the envelope further in this field, some notable gaps remain, particularly the lack of large-scale, in-the-wild training data and fully quantized models which are optimized for live translation. Most prior systems rely on a clean and controlled dataset rather than real-world videos, and only a small subset of research experimentally deploys aggressively quantized models into practical applications. Our project seeks to address these gaps by integrating MediaPipe landmark extraction, automated web scraping of diverse YouTube signing videos, and low-bit integer quantization to create a compact, robust, real-time sign-to-text system. By being able to focus simultaneously on data diversity, lightweight model design, and deployment efficiency, our work looks to contribute a unique combination not yet represented in existing SLR research. 

## System Overview
Our system is designed as an end to end pipeline that recognizes ASL signs from video and moves towards real time interaction. Conceptually, it has three layers: data acquisition and preprocessing, sequence based sign recognition, and an interactive real time prototype. 

### Application Context and Impact:
The primary motivation behind this system is to reduce communication friction between DHH signers and hearing non-signers in everyday situations where a human interpreter is unavailable. Examples include short, transactional interactions, such as asking for directions or clarifying a simple question, where it is impractical to schedule a professional interpreter. 

By recognizing ASL signs directly from video without requiring specialized equipment, our system aims to move towards low cost, camera based assistance tools that could run on commodity hardware. Even partial recognition, such as fingerspelled words, common phrases, or domain specific vocabulary, can help non signers infer more quickly, and communicate with DHH signers in speech dominated spaces. 

### Target Users and Use Cases:
This system supports three main categories of stakeholders. Primary users, or DHH signers, who are fluent in ASL but frequently encounter settings where no interpreter is present, could use this system to convert to text for a non-signer to read. Secondary users, or hearing non-signers, who do not know ASL but need to communicate with signers such as front desk staff, teachers, or customer service workers, would use this as a visual “subtitle generator” to see on screen text translations. The final category of stakeholders would be institutional users such as universities and clinics, where this system would serve as a supplement to existing accessibility measures for short, low stakes interactions. Based on those users, a sample use scenario would be a check in kiosk where a DHH student signs a few key words (such as “appointment”, “advisor”, “financial aid”), and the system displays recognized terms to the staff member. 

## Method
### Data Retrieval:
This project required a substantial amount of ASL video data to effectively train our model. Fortunately, we were able to source this data from Microsoft’s MS-ASL dataset. The dataset is provided in JSON format, containing YouTube links to ASL videos along with metadata such as the sign being performed and its corresponding start and end timestamps.
To collect and prepare the data, we developed a custom Python script that iterated through the JSON file, downloaded each YouTube video using the YT-DLP package, trimmed the clips to the precise time ranges, and organized them into properly labeled folders. Each sign category was assigned its own directory, and because multiple samples existed for each sign, many folders contained several video clips.
Throughout the data collection process, we faced several challenges: some videos had been made private, others had been removed from YouTube entirely, and after a period of time, changes in YouTube’s policies caused our scraping workflow to break. Despite these obstacles, we collected as much of the dataset as possible for use in this project.

## Experiments and Evaluation

## Limitations and Future Work

## Individual Reflection

## Ethical and Social Considerations
The most substantial risk that our system faces is misrecognition and risk of harm. Automatic sign recognition is inherently imperfect and reflects the nature of human ASL. Misclassified signs can lead to misunderstandings, especially if users over trust the system. In high stakes settings such as medical, legal, or emergency, an incorrect translation could cause real harm to Deaf/Hard of Hearing (DHH) individuals. We could mitigate this risk by communicating uncertainty through confidence scores and/or top-k candidate signs to make ambiguity visible to users.

A second ethical consideration is that public ASL datasets may be biased toward particular demographics, signing styles, lighting conditions, or backgrounds. A model trained on such data may work well on “studio like” videos but perform worse for signers with different skin tones, signing speeds, or camera setups. This raises fairness concerns as some users may consistently receive lower quality recognition than others. In order to mitigate this, we will apply diverse evaluation where possible to evaluate performance across different subsets such as signers, backgrounds, or recording conditions to identify systematic gaps rather than only reporting global accuracy. For longer-term work, we propose collecting more diverse, community-informed datasets and including fairness audits as part of the evaluation protocol.
Our system operates on video data of people signing, potentially capturing identifiable faces, environments, and bystanders. Even in a classroom prototype, storing or transmitting raw video raises privacy concerns. Additionally, using online videos for training must respect the original creators’ expectations and dataset licenses. To account for this, we use local processing by default. We design the pipeline so that inference can run entirely on the local machine without uploading video frames to external servers. Any future user facing deployment should clearly inform signers when the camera is recording and how their data is used, with the option to opt out and delete local recordings. We also restrict training to curated datasets where usage for research and model training is explicitly permitted, rather than arbitrarily scraping personal videos without consent.



## Works Cited
[1] H. R. V. Joze and O. Koller, “MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language,” Proc. British Machine Vision Conf. (BMVC), 2018.

[2] D. Li, C. Wang, X. Zhang, and J. Yang, “Word-Level Deep Sign Language Recognition from Video: A New Large-Scale Dataset and Methods Comparison,” Proc. IEEE Winter Conf. on Applications of Computer Vision (WACV), 2020.

[3] Y. Zhang et al., “Recent Advances on Deep Learning for Sign Language Recognition,” Computer Speech & Language, 2024.

[4] S. Subburaj, F. Nawaz, and S. Bhargava, “Survey on Sign Language Recognition in Context of Vision-Based Technology,” Journal of King Saud Univ. – Computer and Information Sciences, 2022.

[5] J. John and B. V. Sherif, “Hand Landmark-Based Sign Language Recognition Using Deep Learning,” in Machine Learning and Autonomous Systems, Springer, 2022.

[6] B. Alsharif et al., “Real-Time American Sign Language Interpretation Using Deep Learning and Keypoint Tracking,” Sensors, 2025.

[7] K. Li, Y. Chen, and H. Xu, “KD-MSLRT: A Lightweight MediaPipe-Based Sign Language Recognition Model Using Knowledge Distillation and 3D-to-1D Encoding,” arXiv preprint, arXiv:2501.02321, Feb. 2025.

[8] J. Sandhya, A. Gopal, and V. Prakash, “Accelerated Low-Power AI System for Indian Sign Language Recognition on FPGA,” Asian Journal of Computer and Technology, 2023.

[9] H. A. Rubaiyeat et al., “BdSLW401: Transformer-Based Word-Level Bangla Sign Language Recognition Using Relative Quantization Encoding,” arXiv preprint, arXiv:2503.02360, Mar. 2025.



#for Transformer model
python utils.py evaluate \
    --model checkpoints/transformer_best.pth \
    --data_dir /path/to/MS-ASL-Test \
    --json /path/to/MSASL_test.json \
    --batch_size 32 \
    --output_dir evaluation_transformer

#for LSTM model
python utils.py evaluate \
    --model checkpoints/lstm_best.pth \
    --data_dir /path/to/MS-ASL-Test \
    --json /path/to/MSASL_test.json \
    --batch_size 32 \
    --output_dir evaluation_lstm

#for TCN model
python utils.py evaluate \
    --model checkpoints/tcn_best.pth \
    --data_dir /path/to/MS-ASL-Test \
    --json /path/to/MSASL_test.json \
    --batch_size 32 \
    --output_dir evaluation_tcn