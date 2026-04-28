# **"Data Engineering for Large Models: Architecture, Algorithms, and Project Practice"**

------

# Book Overview

- [**Preface**](preface.md)
- **Part 1: Overview & Infrastructure** (Building Data-Centric AI Cognition and High-Performance Data Foundation)
- **Part 2: Text Pre-training Data Engineering** (From Massive Noise to High-Quality Corpus)
- **Part 3: Multimodal Data Engineering** (Collection, Cleaning & Alignment for Image, Video & Audio)
- **Part 4: Instruction Fine-tuning & Preference Data** (Making Models Follow Instructions and Learn Preferences)
- **Part 5: Synthetic Data Engineering** (Breaking Through Human Data Bottlenecks)
- **Part 6: Reasoning & Agent Data Engineering** (Chain-of-Thought, Tool Calling & Multi-turn Interaction)
- **Part 7: Application-Level Data Engineering** (RAG & Online Feedback Loop)
- **Part 8: DataOps & Platform Engineering** (DataOps Flywheel & Platform Observability)
- **Part 9: Privacy, Compliance & Data Security** (Data Governance & Privacy-Enhancing Technologies)
- **Part 10: Capstone Projects** (10 End-to-End Capstone Projects)

------

# Detailed Outline

## Part 1: Overview & Infrastructure

> **Goal:** Establish a systematic Data-Centric AI mindset and build a high-performance data processing environment with cost governance.

### Chapter 1: Data Revolution in the LLM Era

- 1.1 **Insights from Scaling Laws:** Data quality > quantity — the paradigm shift from "big data" to "high-quality data."
- 1.2 **LLM Data Lifecycle:** Pre-training → SFT → RLHF → RAG.
- 1.3 **Challenges & Opportunities:** The interplay of heterogeneous multimodality, copyright compliance, and compute costs.

### Chapter 2: LLM Data Lifecycle & Quality Assessment Framework

- 2.1 **Full Data Lifecycle View:** The closed loop of Collection → Cleaning → Annotation → Iteration.
- 2.2 **Multi-dimensional Data Quality Assessment:** A quantitative framework for accuracy, diversity, coverage, and compliance.
- 2.3 **Quality-driven Iteration:** How to use evaluation metrics to guide data engineering decisions.

### Chapter 3: AI-Native Data Stack & Cost Governance

- 3.1 **AI-Native Data Stack:**
  - Storage: Object storage (S3/MinIO) vs Data lakes (Iceberg/Hudi).
  - Compute: Spark vs **Ray Data** vs **Dask** — three distributed frameworks compared.
  - Vector Databases: Milvus / Qdrant / Weaviate / Pinecone selection and QPS vs Recall tradeoffs.
- 3.2 **Data Formats & I/O Optimization:**
  - Parquet vs JSONL vs WebDataset (multimodal scenarios).
  - Compression algorithms and read performance optimization; GPU training I/O bottleneck strategies.
- 3.3 **Cost Governance:** Full-chain cost accounting and optimization strategies for compute, storage, and annotation.

------

## Part 2: Text Pre-training Data Engineering

> **Goal:** Process massive unstructured text to build the model's linguistic cognitive foundation.

### Chapter 4: Data Sources, Acquisition & Copyright

- 4.1 **Deconstructing Open-source Datasets:** Deep analysis of Common Crawl, C4, RefinedWeb, The Pile, and DCLM.
- 4.2 **High-performance Crawling Systems:** `Trafilatura` parsing library and distributed crawler architecture design.
- 4.3 **Specialized Data Acquisition:** Extraction strategies for code (GitHub), papers (ArXiv/S2ORC), and book data.
- 4.4 **Copyright Compliance:** Open-source license identification, robots.txt compliance, and data authorization management.

### Chapter 5: Cleaning, Deduplication & Decontamination

- 5.1 **Heuristic Filtering Rules:** Language identification (FastText), perplexity filtering, length and punctuation distribution.
- 5.2 **Large-scale Deduplication (Exact vs Fuzzy):**
  - **Exact Deduplication:** Hash methods for rapidly removing identical documents.
  - **Fuzzy Deduplication:** MinHash LSH algorithm principles and distributed implementation.
- 5.3 **Privacy Cleaning (PII Removal):** Using Presidio to identify and mask emails, IPs, phone numbers, and addresses.
- 5.4 **Benchmark Decontamination:** Ensuring training data doesn't contain test-set questions from GSM8K, MMLU, etc.

### Chapter 6: Tokenization, Serialization & Efficient Loading

- 6.1 **Tokenizer Principles:** BPE, WordPiece, Unigram and Byte-Level BPE in depth.
- 6.2 **Efficient Vocabulary Construction:** Domain-specific vocabulary expansion and Chinese vocabulary extension engineering.
- 6.3 **Data Mixing:** Dynamic sampling strategies and Curriculum Learning data arrangement.
- 6.4 **High-performance DataLoader:** Multi-process prefetching, memory mapping, and GPU direct-transfer optimization.

### Chapter 7: Data Evaluation, Quality Loop & Operational Iteration

- 7.1 **Model-based Quality Scoring:** Using fastText/BERT for "textbook-quality" scoring.
- 7.2 **Data Recipe:** Experiments and optimization of mixing ratios from multiple sources.
- 7.3 **Quality Flywheel:** A data iteration loop driven by evaluation result write-back.

------

## Part 3: Multimodal Data Engineering

> **Goal:** Process images, video, and audio to support training of GPT-4V/Sora-class models.

### Chapter 8: Image-Text Pair Data Engineering

- 8.1 **Data Paradigms:** Image-text pairs (LAION-5B) vs Interleaved documents (OBELICS/MMC4).
- 8.2 **Image Acquisition & Preprocessing:** `img2dataset` high-concurrency downloading, GPU-accelerated decoding (NVIDIA DALI).
- 8.3 **Multimodal Cleaning Pipeline:**
  - **Aesthetic Scoring:** Using CLIP-Score to filter high-aesthetic images.
  - **Image-text Alignment Filtering:** Removing samples where descriptions don't match images.
  - **Safety Detection:** NSFW and watermark detection.

### Chapter 9: Recaptioning & Document Understanding

- 9.1 **Limitations of Alt-text:** Why raw web descriptions are unusable.
- 9.2 **Synthetic Caption Factory:** Using BLIP-2 / LLaVA / CogVLM to regenerate detailed captions; prompt strategies to control granularity.
- 9.3 **OCR Enhancement & Document Understanding:** Extracting in-image text and fusing it into descriptions; supporting document image understanding tasks.

### Chapter 10: Video & Audio Data Engineering

- 10.1 **Video Processing Pipeline:** Scene detection and keyframe extraction strategies.
- 10.2 **Video Tokenization:** Video compression and discrete representation.
- 10.3 **Audio Alignment:** Large-scale ASR with Whisper and Force Alignment (timestamp alignment).

### Chapter 11: Cross-Modal Alignment & Fusion

- 11.1 **Alignment Goals:** Technical paths and evaluation criteria for vision-language alignment.
- 11.2 **Cross-modal Data Mixing Strategies:** Experiments on image-text / video / audio mixing ratios.
- 11.3 **Multimodal Data Quality Assessment:** An automated evaluation framework using CLIP Score, semantic consistency, and dialogue coherence.

------

## Part 4: Instruction Fine-tuning & Preference Data

> **Goal:** Make models follow instructions and learn human preferences; build high-quality alignment data.

### Chapter 12: SFT Data Design & Instruction Taxonomy

- 12.1 **Prompt Engineering for Data Production:** Writing robust System Prompts.
- 12.2 **Automated Construction Methods:** Self-Instruct for leveraging strong models to generate instructions; Evol-Instruct for evolving instruction complexity.
- 12.3 **Chain-of-Thought (CoT) Data:** Constructing step-by-step reasoning samples.
- 12.4 **Instruction Diversity & Coverage:** Balancing task type distribution and designing difficulty gradients.

### Chapter 13: Preference Data & Reward Signals

- 13.1 **Preference Data Format:** Constructing chosen vs rejected sample pairs and DPO data standards.
- 13.2 **RLAIF (AI Feedback):** Using LLMs to replace human preference scoring; Constitutional AI methodology.
- 13.3 **Reward Model Training Data:** RM data collection strategies and quality control.

### Chapter 14: Annotation Platforms, QA Systems & Data Operations

- 14.1 **Annotation Platform Selection:** Label Studio, Scale AI, Labelbox comparison and enterprise custom solutions.
- 14.2 **Annotation Consistency (IAA):** Computing and controlling quality with Cohen's Kappa and Fleiss' Kappa.
- 14.3 **Crowdsourcing Management:** Annotator screening, training, and quality-inspection sampling.
- 14.4 **Data Operations:** Requirements prioritization, delivery SLA, and issue resolution processes.

------

## Part 5: Synthetic Data Engineering

> **Goal:** Break through human data bottlenecks by using models to generate high-quality training data.

### Chapter 15: Synthetic Data Factory: From Seeds to Validation

- 15.1 **Textbook-quality Data (Textbooks Are All You Need):** Methodology for synthesizing high-quality domain knowledge.
- 15.2 **Code & Math Synthesis:** PoT (Program of Thought) generates code and executes it, verifying data correctness via execution results.
- 15.3 **Multimodal Instruction Synthesis:** Using GPT-4o to construct complex image-based reasoning Q&A.
- 15.4 **Synthetic Data Validation:** Automated validation pipelines and manual sampling strategies.

### Chapter 16: Knowledge Distillation & Model Collaboration

- 16.1 **Distillation Data Paradigm:** Data strategies for transferring capabilities from strong (Teacher) to weak (Student) models.
- 16.2 **Collaborative Generation:** Multi-model voting, best-of-N, and Critic model filtering.
- 16.3 **Self-improvement:** The closed loop of model generation → validation → filtering → retraining.

### Chapter 17: Synthetic Data Quality Control & Model Collapse

- 17.1 **Synthetic Data Quality Detection:** Diversity analysis, distribution shift detection, and hallucination rate evaluation.
- 17.2 **Model Collapse:** Cause analysis and engineering strategies to prevent overfitting on synthetic data.
- 17.3 **Real vs Synthetic Data Mixing:** Optimal ratio experiments for mixed training.

------

## Part 6: Reasoning & Agent Data Engineering

> **Goal:** Build high-quality data supporting complex reasoning, tool calling, and multi-turn interaction.

### Chapter 18: Chain-of-Thought & Reasoning Data Engineering

- 18.1 **CoT Data Types:** Zero-shot CoT, Few-shot CoT, and automated CoT generation.
- 18.2 **Process Reward Data (PRM):** Building step-by-step reward-annotated math reasoning datasets.
- 18.3 **Reasoning Data Validation:** Automated correctness checks based on code execution and formal verification.

### Chapter 19: Tool-Use & Function-Calling Data

- 19.1 **Tool-Use Data Format:** Function calling data specifications and schema design.
- 19.2 **Tool-Call Trajectory Generation:** Automated API call chain collection and trajectory annotation.
- 19.3 **Multi-tool Collaboration Scenarios:** Construction strategies for multi-step tool-call data in complex tasks.

### Chapter 20: Agent Memory & Multi-turn Interaction Data

- 20.1 **Conversation History Management:** Collection, filtering, and formatting of long-context dialogue data.
- 20.2 **Memory Mechanism Data:** Data engineering for short-term memory, long-term memory, and external knowledge base interaction.
- 20.3 **Multi-agent Interaction Data:** Collection and quality control of agent collaboration trajectories.

------

## Part 7: Application-Level Data Engineering

> **Goal:** Enterprise-oriented solutions for external knowledge base parsing, retrieval, and feedback loops.

### Chapter 21: RAG Data Pipeline

- 21.1 **Deep Document Parsing:** Complex PDF processing (table reconstruction, multi-column recognition), `Unstructured`/`LlamaParse` in practice.
- 21.2 **Chunking Strategies:** Semantic chunking, recursive chunking, and Parent-Child Indexing.
- 21.3 **Vectorization & Storage:** Embedding model fine-tuning and vector database optimization.

### Chapter 22: Multimodal RAG & Visual Retrieval

- 22.1 **Cross-modal Retrieval:** Using CLIP/SigLIP for text-to-image and image-to-text search.
- 22.2 **ColPali Architecture in Practice:** Vision-language model based document retrieval (skip OCR, directly understand document images).
- 22.3 **Multi-route Recall & Reranking:** Hybrid retrieval strategies and training data construction for reranking models.

### Chapter 23: Online Feedback Loop & Knowledge Update

- 23.1 **User Feedback Collection:** Quantifying explicit feedback (likes/dislikes) and implicit feedback (dwell time, copy behavior).
- 23.2 **Continual Learning Data Strategies:** Incremental updates and mitigation solutions for catastrophic forgetting.
- 23.3 **Knowledge Freshness Management:** The knowledge cutoff problem and update mechanisms for time-sensitive data.

------

## Part 8: DataOps & Platform Engineering

> **Goal:** Systematize data engineering for sustainable operation; build the DataOps flywheel and observable platform.

### Chapter 24: DataOps Flywheel & Team Organization

- 24.1 **DataOps Flywheel:** The four-wheel drive mechanism of requirements pool, data pool, experiment pool, and issue pool.
- 24.2 **Team Roles & Collaboration:** Division of labor among data engineers, annotation engineers, quality evaluators, and data owners; RACI matrix.
- 24.3 **Cadence Design:** Team rhythm with Monday requirements sync, Wednesday quality inspection, and Friday delivery retrospective.

### Chapter 25: Data Versioning & Experiment Tracking

- 25.1 **Five-level Version Granularity:** A versioning system at sample, shard, dataset, experiment, and release-package levels.
- 25.2 **Toolchain:** Use cases and integration solutions for DVC, LakeFS, and MLflow.
- 25.3 **Experiment Cards:** Complete experiment tracking field design and value mining from failed experiments.
- 25.4 **Data Lineage Graph:** Three query perspectives — forward tracking, reverse tracking, and diff comparison.

### Chapter 26: Data Platform Observability

- 26.1 **Three-tier Metrics System:** A layered monitoring framework for task metrics, quality metrics, and business metrics.
- 26.2 **Alerting Strategy:** A four-level alert system (P0–P3) and a four-step decision tree for anomaly attribution.
- 26.3 **Operations Dashboard:** Three-dimensional design of platform health view, data quality view, and business operations view.

------

## Part 9: Privacy, Compliance & Data Security

> **Goal:** Build a compliance governance framework for LLMs; ensure safe and compliant data use.

### Chapter 27: Data Compliance Framework & Governance

- 27.1 **Data Compliance Challenges:** Interpreting core requirements of GDPR, CCPA, and China's Data Security Law.
- 27.2 **Compliance Governance Framework:** Data classification, access control, and end-to-end audit mechanisms.
- 27.3 **PII Identification & Anonymization:** A large-scale pipeline for processing personal information in training data.

### Chapter 28: Federated Learning & Privacy-Enhancing Technologies

- 28.1 **Federated Learning Principles:** Application scenarios for horizontal, vertical, and federated transfer learning.
- 28.2 **Differential Privacy (DP):** Implementing DP-SGD in LLM training and privacy budget management.
- 28.3 **Secure Multi-party Computation (MPC) & Homomorphic Encryption:** Cryptographic protection solutions for collaborative data training.

------

## Part 10: Capstone Projects

> **Goal:** Through 10 end-to-end projects, connect all technical topics from the book and provide a runnable codebase.

### Project 1: Building a Distributed Mini-C4 Pipeline with Ray

- **Scenario:** From Common Crawl raw data (WARC) to high-quality Parquet data.
- **Core Technologies:** Trafilatura parsing, Ray distributed MinHash deduplication, KenLM quality filtering.
- **Output:** Cleaned plain text corpus and processing pipeline.

### Project 2: Domain Expert SFT (Legal)

- **Scenario:** Building industry expert fine-tuning data from unstructured PDF documents.
- **Core Technologies:** Self-Instruct instruction construction, CoT reasoning enhancement, data diversity balancing.
- **Output:** `domain_expert.jsonl` instruction fine-tuning dataset.

### Project 3: LLaVA Multimodal Instruction Data Factory

- **Scenario:** Training a multimodal model that can understand images.
- **Core Technologies:** GPT-4o API for generating multi-turn image-text dialogues, Bounding Box data alignment, multi-image interleaved format processing.
- **Output:** Image-text dataset with visual instructions.

### Project 4: Synthetic Math & Code Textbook Factory

- **Scenario:** Improving small model logical reasoning capabilities.
- **Core Technologies:** Evol-Instruct evolutionary strategies, Python code execution sandbox (Sandbox) verification, PoT data formatting.
- **Output:** Verified high-quality synthetic reasoning dataset.

### Project 5: Multimodal RAG Financial Report Assistant

- **Scenario:** Retrieving and answering questions about annual reports containing complex charts.
- **Core Technologies:** PDF table and chart parsing, multi-route recall (hybrid retrieval), ColPali visual retrieval.
- **Output:** A RAG knowledge base system supporting chart Q&A.

### Project 6: CoT Reasoning Dataset Construction & PRM Training

- **Scenario:** Building math reasoning process data and training a Process Reward Model.
- **Core Technologies:** Automated CoT generation, step-level annotation, PRM training data pipeline.
- **Output:** Reasoning process dataset and PRM model.

### Project 7: Agent Tool-Use Data Factory

- **Scenario:** Building Agent training data for multi-tool calling.
- **Core Technologies:** Automated tool-call chain collection, trajectory annotation, multi-step task data validation.
- **Output:** Agent tool-calling training dataset.

### Project 8: Enterprise DataOps Platform: From Data Projects to Org-Level Governance

- **Scenario:** Platform-level realization from data projects to organization-level governance capabilities.
- **Core Technologies:** Airflow scheduling, DVC version management, Great Expectations quality monitoring.
- **Output:** Enterprise-grade data operations system and platform architecture plan.

### Project 9: Privacy-Preserving Data Pipeline

- **Scenario:** Building a training data processing system that meets compliance requirements.
- **Core Technologies:** Federated learning framework, differential privacy DP-SGD, automated PII anonymization pipeline.
- **Output:** Compliant training data pipeline and privacy audit report.

### Project 10: End-to-End LLM Data Flywheel

- **Scenario:** Online feedback-driven continuous data iteration and model improvement.
- **Core Technologies:** User feedback collection, automated annotation, incremental training, and effectiveness evaluation loop.
- **Output:** End-to-end LLM data flywheel system.
