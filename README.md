# SynData-Survey

## 🤗Papers that we read
| 類別          | who are you | paper 題目                                                                                                          | year | paper 連結                                            | 技術類型（四類）                         | application                 | dataset  | 一句話總結                                                                                                                                                                                                               | 
|-------------|-------------|-------------------------------------------------------------------------------------------------------------------|------|-----------------------------------------------------|----------------------------------|-----------------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| text        | 沛妤          | Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks           | 2021 | https://arxiv.org/abs/2010.08240                    | pre-train and fine-tune          | Natural language inference  | quantity | 用 cross-encoders 標記資料來增強 Bi-encoders 模型                                                                                                                                                                             | 沛妤_10 papers               | 沛妤_10 papers               |                |
| text        | 沛妤          | AugGPT: Leveraging ChatGPT for Text Data Augmentation                                                             | 2023 | https://arxiv.org/abs/2302.13007                    | "pre-train, prompt, and predict" | Text classification         | quantity | 透過 GPT 擴充舊資料 + 新資料 => 希望在新資料上表現好                                                                                                                                                                                    |                            |                            |                |
| text        | 沛妤          | EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks                      | 2019 | https://arxiv.org/abs/1901.11196                    | feature engineering              | Text classification         | quantity | 就是很直覺的四種調換文字的方式來 aug，可以當 baseline                                                                                                                                                                                   |                            |                            |                |
| text        | 沛妤          | AEDA: An Easier Data Augmentation Technique for Text Classification                                               | 2021 | https://arxiv.org/abs/2108.13230                    | feature engineering              | Text classification         | quantity | 在原始文本中隨機插入標點符號，比 EDA 好                                                                                                                                                                                              |                            |                            |                |
| text        | 沛妤          | Text Smoothing: Enhance Various Data Augmentation Methods on Text Classification Tasks                            | 2022 | https://arxiv.org/pdf/2202.13840.pdf                | architecture engineering         | Text classification         | quantity | text smoothing 學習 token 的機率分佈，而不僅是它是否出現                                                                                                                                                                             |                            |                            |                |
| text        | 沛妤          | AUGNLG: Few-shot Natural Language Generation using Self-trained Data Augmentation                                 | 2021 | https://arxiv.org/pdf/2106.05589.pdf                | architecture engineering         | Text Generation             | quantity | 檢索關鍵字匹配 -> 過濾領域無關 -> 合成標記數據                                                                                                                                                                                         |                            |                            |                |
| text        | 沛妤          | End-to-End Synthetic Data Generation for Domain Adaptation of Question Answering Systems                          | 2020 | https://arxiv.org/pdf/2010.06028.pdf                | architecture engineering         | Question answering          | quantity | 將段落輸入 encoder 中，decoder 逐詞生成問題和答案，並使用生成中的值作為過濾分數篩選生成數據                                                                                                                                                              |                            |                            |                |
| text        | 沛妤          | GDA: Generative Data Augmentation Techniques for Relation Extraction Tasks                                        | 2023 | https://arxiv.org/pdf/2305.16663.pdf                | pre-train and fine-tune          | Text Generation             | quality  | one encoder 預測原始句、one encoder 生成相似句法的目標句                                                                                                                                                                            |                            |                            |                |
| text        | 沛妤          | Do Not Have Enough Data? Deep Learning to the Rescue!                                                             | 2019 | https://arxiv.org/pdf/1911.03118.pdf                | "pre-train, prompt, and predict" | Text classification         | quantity | 利用 GPT-2 來進行 data aug，從而在 few-shot 場景下達到很好的增強效果                                                                                                                                                                     |                            |                            |                |
| text        | 沛妤          | Training Question Answering Models From Synthetic Data                                                            | 2020 | https://arxiv.org/abs/2002.09599                    | "pre-train, prompt, and predict" | Question answering          | quantity | 答案生成 -> 問題生成 -> roundtrip filtration                                                                                                                                                                                |                            |                            |                |
| text        | QQ          | Adapting Neural Machine Translation with Parallel Synthetic Data                                                  | 2017 | https://aclanthology.org/W17-4714/                  | feature engineering              | Translation                 | quantity | 透過cosine similarity 輔助計算，找到和target domain相關的句子進行資料的擴充並進行翻譯來擴充 data                                                                                                                                                  | 心瑜_文件                      | 心瑜_簡報                      | 10 篇真的是太多了吧！！！ |
| text        | QQ          | MulDA: A Multilingual Data Augmentation Framework for Low-Resource Cross-Lingual NER                              | 2021 | https://aclanthology.org/2021.acl-long.453/         | pre-train and fine-tune          | Translation                 | quantity | 首先通過訓練 LSTM-LM來產生特定語言的synthetic data，再透過 mBART 搭配 mask 產生不同 diversity 的synthetic data                                                                                                                               |                            |                            |                |
| text        | QQ          | XLA: A Robust Unsupervised Data Augmentation Framework for Cross-Lingual NLP                                      | 2021 | https://openreview.net/forum?id=w5uur-ZwCXn         | pre-train and fine-tune          | Translation                 | quantity | XLA是一個 robust unsupervised cross-lingual augmentation framework 針對 cross-lingual generalization的LM來增加 data                                                                                                          |                            |                            |                |
| text        | QQ          | Data Augmentation for Abstractive Query-Focused Multi-Document Summarization                                      | 2021 | https://arxiv.org/abs/2103.01863                    | feature engineering              | Summarizing                 | quantity | 建立了2個  query - summary dataset ，一個是有真實的 summary 建立虛擬的 query，一個是有真實的 query 建立虛擬的 summary                                                                                                                             |                            |                            |                |
| text        | QQ          | Paraphrase Augmented Task-Oriented Dialog Generation                                                              | 2020 | https://aclanthology.org/2020.acl-main.60/          | pre-train and fine-tune          | Dialogue                    | quantity | 首先做Paraphrase Data augmentation然後再用encoder-decoder 生成Response                                                                                                                                                       |                            |                            |                |
| text        | QQ          | Transforming Wikipedia into Augmented Data for Query-Focused Summarization                                        | 2022 | https://arxiv.org/abs/1911.03324                    | feature engineering              | Summarizing                 | quantity | 透過WIKIPEDIA 建立一個 query-summarization 的 Dataset                                                                                                                                                                      |                            |                            |                |
| text        | QQ          | Data Augmentation using Pre-trained Transformer Models                                                            | 2021 | https://arxiv.org/abs/2003.02245                    | pre-train and fine-tune          | Text classification         | quantity | 使用三種不同的LM來做Data augmentation在三種不同的task上，看看效果如何                                                                                                                                                                      |                            |                            |                |
| text        | QQ          | Mitigating Class Imbalance in Sentiment Analysis through GPT-3-Generated Synthetic Sentences                      | 2023 | https://www.mdpi.com/2076-3417/13/17/9766           | pre-train and fine-tune          | Text classification         | quality  | 當分類問題的訓練資料集中出現數量不平衡的問題時，使用GPT-3做Data augmentation來將 Dataset 變成平衡的 Data                                                                                                                                              |                            |                            |                |
| text        | QQ          | DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows                                  | 2024 | https://arxiv.org/abs/2402.10379                    | "pre-train, prompt, and predict" | Text Generation             | quantity | 一個方便使用LM的工具，如果我們要做實驗的話可以考慮試試看這個工具                                                                                                                                                                                   |                            |                            |                |
| text        | QQ          | Synthetic Dialogue Dataset Generation using LLM Agents                                                            | 2024 | https://arxiv.org/abs/2401.17461                    | "pre-train, prompt, and predict" | Dialogue                    | quantity | Dual LLM 使用 prompt engineering 來模擬對話，產生 dialog data                                                                                                                                                                 |                            |                            | 終於看完了！！！       |
| text        | Victoria    | WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation                                | 2022 | https://arxiv.org/pdf/2201.05955.pdf                | "pre-train, prompt, and predict" | Natural language inference  | quantity | 以gpt3使用MultiNLI dateset 產生相對或類似的幾個句子，過濾後由人類evaluate                                                                                                                                                                 | Victoria                   | Victoria_簡報                |                |
| text        | Victoria    | Data Augmentation for Intent Classification with Off-the-shelf Large Language Models                              | 2022 | https://arxiv.org/pdf/2204.01959.pdf                | "pre-train, prompt, and predict" | Text classification         | quality  | 使用GPT3以few shot方式產生intent classification data                                                                                                                                                                       |                            |                            |                |
| text        | Victoria    | GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation                                             | 2021 | https://arxiv.org/pdf/2104.08826.pdf                | "pre-train, prompt, and predict" | Text classification         | quantity | 利用GPT3即pseudo labeling的方式擴增資料                                                                                                                                                                                       |                            |                            |                |
| text        | Victoria    | Character-level Convolutional Networks for Text Classification                                                    | 2016 | https://arxiv.org/pdf/1509.01626.pdf                | feature engineering              | Text classification         | quantity | 選擇利用它們的同義詞替換單詞或短語，library來自LibreOffice                                                                                                                                                                              |                            |                            |                |
| text        | Victoria    | Generative Data Augmentation for Commonsense Reasoning                                                            | 2020 | https://aclanthology.org/2020.findings-emnlp.90.pdf | "pre-train, prompt, and predict" | Question answering          | quality  | 分別fine tune 兩個LM，一個是Generator(GPT2)，一個是organic training(RoBERTa)，對generator產生的data做relabeling，再做influence和diversity filtering                                                                                       |                            |                            |                |
| text        | Victoria    | Data Augmentation for Low-Resource Neural Machine Translation                                                     | 2017 | https://arxiv.org/pdf/1705.00440.pdf                | architecture engineering         | Translation                 | quality  | 翻譯資料擴增（TDA），它通過改變平行語料中現有的句子來擴增訓練資料，與電腦視覺中的資料擴增方法相似                                                                                                                                                                  |                            |                            |                |
| text        | Victoria    | Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations                                   | 2018 | https://arxiv.org/pdf/1805.06201.pdf                | architecture engineering         |                             | quantity | 提出了上下文增強，用於增強單詞與更多不同的單詞。不使用同義詞，而是使用根據原始單詞所處上下文預測出的單詞進行增強。                                                                                                                                                           |                            |                            |                |
| text        | Victoria    | Few-shot learning through contextual data augmentation                                                            | 2021 | https://aclanthology.org/2021.eacl-main.90.pdf      | "pre-train, prompt, and predict" | Translation                 | quantity | 提出Contextual data augmentation的方法                                                                                                                                                                                   |                            |                            |                |
| text        | Victoria    | Improving Few-shot Generalization of Safety Classifiers via Data Augmented Parameter-Efficient Fine-Tuning        | 2023 | https://arxiv.org/pdf/2310.16959.pdf                | "pre-train, prompt, and predict" | Text classification         | quality  | 提出DAPT，比較 prompt tuning(PEFT) 和 ICL (in-context learning) ，認為PEFT較好                                                                                                                                                 |                            |                            |                |
| text        | Victoria    | FlipDA: Effective and Robust Data Augmentation for Few-Shot Learning                                              | 2022 | https://aclanthology.org/2022.acl-long.592.pdf      | "pre-train, prompt, and predict" | Text classification         | quality  | 利用T5產生正反兩面的候選人，輸入進classifier，若輸出label不一致則刪掉                                                                                                                                                                         |                            |                            |                |
| image       | 周敦翔         | CamDiff: Camouflage Image Augmentation via Diffusion Model                                                        | 2023 | https://arxiv.org/abs/2304.05469                    | "pre-train, prompt, and predict" | object detection            | quality  | Augment existing camouflage object detection (COD) dataset with salient objects to enhance the robustness of COD models.                                                                                            | Image Augmentation 敦翔      | Image Augmentation 敦翔      |                |
| image       | 周敦翔         | Generating images of rare concepts using pre-trained diffusion models                                             | 2023 | https://arxiv.org/abs/2304.14530                    | pre-train and fine-tune          | image classification        | quantity | Rare concepts can be correctly generated by carefully selecting suitable generation seeds in the noise space                                                                                                        |                            |                            |                |
| image       | 周敦翔         | Diversify Your Vision Datasets with Automatic Diffusion-Based Augmentation                                        | 2023 | https://arxiv.org/abs/2305.16289                    | "pre-train, prompt, and predict" | image classification        | quantity | Augment the training data via descriptions of the domains seen in training data and language-guided image editing                                                                                                   |                            |                            |                |
| image       | 周敦翔         | Semantic Generative Augmentations for Few-Shot Counting                                                           | 2023 | https://arxiv.org/abs/2311.16122                    | pre-train and fine-tune          | object detection            | quantity | "Synthesize few-shot object counting data with stable diffusion, conditioned on a textual prompt and a density map"                                                                                                 |                            |                            |                |
| image       | 周敦翔         | Diffusion-based Data Augmentation for Nuclei Image Segmentation                                                   | 2024 | https://arxiv.org/abs/2310.14197                    | architecture engineering         | semantic segmentation       | quantity | Introduce a two-step strategy for diffusion-based nuclei segmentation augmentation                                                                                                                                  |                            |                            |                |
| image       | 周敦翔         | GPT-Prompt Controlled Diffusion for Weakly-Supervised Semantic Segmentation                                       | 2023 | https://arxiv.org/abs/2310.09760                    | "pre-train, prompt, and predict" | semantic segmentation       | quantity | "Utilizes GPT with image labels to generate diverse prompts, and use the diffusion model to synthesize images. "                                                                                                    |                            |                            |                |
| image       | 周敦翔         | Towards Generalizable Tumor Synthesis                                                                             | 2024 | https://arxiv.org/abs/2402.19470                    | pre-train and fine-tune          | semantic segmentation       | quantity | Diffusion Models can create realistic tumors generalized to a range of organs even when trained on tumor examples from only one organ.                                                                              |                            |                            |                |
| multi-modal | 周敦翔         | VIXEN: Visual Text Comparison Network for Image Difference Captioning                                             | 2024 | https://arxiv.org/abs/2402.19119                    | "pre-train, prompt, and predict" | image difference captioning | quantity | Training on synthetically manipulated images to improve image difference captioning                                                                                                                                 |                            |                            |                |
| image       | 周敦翔         | ShapeBoost: Boosting Human Shape Estimation with Part-Based Parameterization and Clothing-Preserving Augmentation | 2024 | https://arxiv.org/abs/2403.01345                    | feature engineering              | human pose estimation       | quantity | Propose a clothing-preserving data augmentation module to generate realistic images with diverse body shapes for human shape recovery.                                                                              |                            |                            |                |
| image       | 周敦翔         | Boosting Dermatoscopic Lesion Segmentation via Diffusion Models with Visual and Textual Prompts                   | 2023 | https://arxiv.org/abs/2310.02906                    | pre-train and fine-tune          | semantic segmentation       | quantity | Adapt diffusion model with lesion-specific visual and textual prompts for generating skin lesion images.                                                                                                            |                            |                            |                |
| image       | 高長聖         | Diversity is Definitely Needed: Improving Model-Agnostic Zero-shot Classification via Stable Diffusion            | 2023 | https://arxiv.org/pdf/2302.03298.pdf                | "pre-train, prompt, and predict" | image classification        | quality  | Enhance the diversity of generated images through different prompts to increase the performance of the image classification task using the diffusion model.                                                         | Synthetic Data Paper Intro | Synthetic Data Paper Intro |                |
| image       | 高長聖         | DALL-E for Detection: Language-driven Compositional Image Synthesis for Object Detection                          | 2022 | https://arxiv.org/abs/2206.09592                    | "pre-train, prompt, and predict" | object detection            | quantity | "Allow DALL-E to generate background and foreground objects separately, then combine them, so that the synthetic data for tasks such as object detection or semantic segmentation are generated"                    |                            |                            |                |
| image       | 高長聖         | DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models      | 2023 | https://arxiv.org/abs/2303.11681                    | "pre-train, prompt, and predict" | semantic segmentation       | quantity | "While generating images using the diffusion model, synthetic semantic segmentation data is simultaneously generated through cross-attention."                                                                      |                            |                            |                |
| image       | 高長聖         | Dataset Diffusion: Diffusion-based Synthetic Dataset Generation for Pixel-Level Semantic Segmentation             | 2023 | https://arxiv.org/abs/2309.14303                    | "pre-train, prompt, and predict" | semantic segmentation       | quantity | Generate synthetic semantic segmentation data containing multiple different classes and utilize self-attention to refine cross-attention maps.                                                                      |                            |                            |                |
| image       | 高長聖         | DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models                                   | 2023 | https://arxiv.org/abs/2308.06160                    | "pre-train, prompt, and predict" | semantic segmentation       | quantity | "By decoding the latent code of the diffusion model, label different tasks, thereby generating synthetic data for different tasks"                                                                                  |                            |                            |                |
| text        | 高長聖         | Self-Instruct: Aligning Language Models with Self-Generated Instructions                                          | 2022 | https://arxiv.org/abs/2212.10560                    | "pre-train, prompt, and predict" | Instruction tuning          | quantity | Develop a framework to leverage the in-context learning capabilities of Large Language Models (LLM) for generating synthetic text data.                                                                             |                            |                            |                |
| multi-modal | 高長聖         | Visual Instruction Tuning                                                                                         | 2023 | https://arxiv.org/abs/2304.08485                    | "pre-train, prompt, and predict" | Instruction tuning          | quantity | Generate a visual instruction tuning dataset through self-instruction.                                                                                                                                              |                            |                            |                |
| multi-modal | 高長聖         | MIMIC-IT: Multi-Modal In-Context Instruction Tuning                                                               | 2023 | https://arxiv.org/abs/2306.05425                    | "pre-train, prompt, and predict" | Instruction tuning          | quantity | "Generate synthetic visual instruction datasets for various domains (image, video, etc.) through the self-instruct method."                                                                                         |                            |                            |                |
| multi-modal | 高長聖         | Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning                                | 2023 | https://arxiv.org/abs/2306.14565                    | "pre-train, prompt, and predict" | Instruction tuning          | quantity | "Enhance the method of visual instruction tuning by incorporating information such as dense captions, object detection, and other details, treating them as image representations in the self-instruction process." |                            |                            |                |
| multi-modal | 高長聖         | DialogCC: Large-Scale Multi-Modal Dialogue Dataset                                                                | 2022 | https://arxiv.org/abs/2212.04119                    | "pre-train, prompt, and predict" | Dialogue                    | quantity | Create a synthetic visual dialogue dataset by leveraging a text-only dialogue dataset and an image captioning dataset                                                                                               |                            |                            |                |

## 🦛Paper Architecture
### 🧸Abstract
### 🦖Introduction
- 解釋 data aug 的總體概念和方法
- 可以用年代去解釋
### 👻Background
- 解釋 LLM、encoder-decoder、diffussion 的概念
### 🐎APPROACHES
- 參考 https://arxiv.org/abs/2107.13586 （用技術分用年代放順序）
  - 🐒feature engineering:
    - text: rule-based
    - image: https://pytorch.org/vision/stable/transforms.html
    - others
  - 🦙architecture engineering: 
    - text: RNN, LSTM, Transformer
    - image: GAN
    - others
  - 🐈pre-train and fine-tune:
    - text: Transformer,  LM(encoder, decoder, BERT,generation-based model, RNN)
    - image: CNN, Diffusion
    - others
  - 🍟pre-train, prompt, and predict:
    - text: Generation LLM(GPT), prompt engineering, context learning
    - image: Diffusion, DALLE
    - others

### 🦘AUGMENTATION OBJECTIVE 

- improve diversity, ex: 少數資源的語言新增
- improve dataset balance ex: label imbalance
- domain shift
> 一個好的 dataset 應該要同時有好的 quality(e.g. 論文目的是為了增加資料的 diversity, availability, accuracy, complete...) 和 足夠的 quantity，可以根據這兩個主題分析 data augmentation 的方式是否有在增加 dataset 的同時也注意 quality

### 🚟APPLICATION
- 📒Text
  - Text classification
  - Question answering
  - Translation
  - Natural language inference
  - Text Generation
  - Summarizing
  - Instruction tuning
  - Others
- Image
  - Image Classification 
  - Semantic Segmentation
  - Object Detection
  - Human Pose Estimation
### Summary
### Conclusion


