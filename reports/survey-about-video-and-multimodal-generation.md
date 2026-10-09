# Deep Research Survey: Video and Multimodal Generation

## TL;DR
- Video and multimodal generation has undergone a profound paradigm shift, transitioning from early GANs and autoregressive models to high-capacity Video Diffusion Models (VDMs) and Spatio-Temporal Diffusion Transformers (STDiTs) [1][2][3][4].
- Foundational breakthroughs established that models could be trained by inflating spatial U-Nets and attention blocks with temporal layers, leveraging zero-shot transfer from text-to-image models without requiring massive paired text-video datasets [5][6][7].
- Recent frontier systems (2023–2025) scale past 13 billion parameters (e.g., HunyuanVideo, CogVideoX, Sora), exhibiting emergent geometric consistency and physical realism while functioning as nascent world simulators [4][8][9].
- Comprehensive benchmarks such as VBench and EvalCrafter reveal critical trade-offs between temporal consistency and motion amplitude, driving innovations in adaptive caching (AdaCache), memory-efficient inference, and robust safety alignment [10][11][12][13].

## Background
The synthesis of realistic video and multimodal content represents one of the most computationally demanding and fast-evolving frontiers in generative artificial intelligence. Historically, video generation relied on Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and autoregressive sequence models [2]. While these classical approaches successfully modeled short, low-resolution clips, they frequently suffered from severe training instability, catastrophic temporal flickering, and limited semantic diversity. 

The introduction of diffusion probabilistic models revolutionized image synthesis, providing a principled score-based framework that soon catalyzed video generation [2]. Early milestones such as Video Diffusion Models (VDM) demonstrated that natural extensions of spatial U-Nets equipped with factorized space-time attention blocks could capture joint image-video distributions [1]. Concurrently, Make-A-Video and Imagen Video established that text-conditioned video synthesis could bypass the extreme scarcity of paired text-video data by transferring powerful aesthetic priors from pretrained text-to-image models and cascading spatial-temporal super-resolution networks [5][14]. These foundational paradigms set the stage for scaling video generators to high-definition outputs and laid the methodological groundwork for modern latent video diffusion and transformer architectures [2].

## Core Architectures and Taxonomies
As video generation matured, architectural designs evolved from factorized 3D U-Nets toward Latent Video Diffusion Models (Video-LDMs) and Spatio-Temporal Diffusion Transformers (STDiTs) [6][3]. Stable Video Diffusion (SVD) established a prominent Video-LDM taxonomy by freezing pretrained text-to-image backbones (e.g., Stable Diffusion 2.1) and inserting dedicated temporal convolution and attention layers after every spatial layer, trained via a multi-stage pipeline encompassing low-resolution video pretraining and high-resolution finetuning [6]. Similarly, AnimateDiff introduced plug-and-play motion modules that enable personalized text-to-image models to generate animated sequences without specialized fine-tuning by merging spatial dimensions into batch axes during temporal self-attention [7].

To overcome the severe computational bottlenecks of global attention across long video token sequences, recent architectures embrace Diffusion Transformers (DiTs) and structured sparsity [3][15]. Frameworks like Open-Sora and Compact Attention exploit spatio-temporal sparsity, factorized attention patterns, and 3D variational autoencoders (VAEs) to maintain global motion coherence while drastically reducing redundant computations across correlated frames [3][15]. Furthermore, multimodal conditioning techniques—such as cross-attention mechanisms for text embeddings, CLIP image embeddings for image-to-video (I2V) tasks, and channel-wise concatenation of noise-augmented conditioning frames—allow models to seamlessly ingest diverse conditioning modalities including text, images, camera trajectories, and audio [6].

## Recent Frontier Advancements (2023–2025)
Between 2023 and 2025, video generation underwent a massive scaling revolution, driven by open-source foundations and proprietary frontier models. HunyuanVideo established a systematic open-source framework for large video generative models exceeding 13 billion parameters, rivaling leading closed-source systems in visual quality and motion fidelity [4]. This was subsequently complemented by lightweight iterations like HunyuanVideo 1.5 and CogVideoX, which leverage expert 3D VAEs and Spatiotemporal Self-Attention (SSTA) mechanisms to ensure robust prompt adherence and open accessibility [16][8].

Concurrently, systems like OpenAI's Sora demonstrated stunning geometrical consistency and 3D spatial awareness, sparking intense research into whether generative video models function as emergent world simulators [9]. However, formal evaluations from a physical law perspective highlight that while these models exhibit near-perfect in-distribution combinatorial generalization, they frequently fail in out-of-distribution physical scenarios—often prioritizing visual aesthetics and color consistency over rigorous Newtonian mechanics or causal gravity [17].

## Applications, Benchmarks, and Trade-offs
Evaluating video and multimodal generation requires sophisticated benchmarking suites that disentangle subjective human preference from objective metrics [10][11]. VBench decomposes video quality into 16 hierarchical dimensions across video quality and video-condition consistency, proving that users consistently prioritize aesthetic appeal and temporal stability over strict lexical text alignment [10]. EvalCrafter further reinforces this through extensive user studies on real-world prompts, emphasizing that current models face persistent bottlenecks in open-world camera control, text rendering, and complex multi-entity interactions [11].

The massive computational footprint of video DiTs—where peak memory can exceed 42GB during iterative denoising—has driven critical optimizations [12][18]. Training-free inference accelerators such as Adaptive Caching (AdaCache) and Streamlined Inference frameworks (incorporating feature slicing, operator group pipelining, and step rehash) achieve up to $4.7\times$ speedups and reduce memory footprints to fit consumer GPUs [12][18]. Finally, to address societal and legal challenges, recent research introduces robust generative watermarking (Safe-Sora) and safety alignment frameworks (SafeVidBench) to safeguard copyright integrity and mitigate adversarial generation risks in video large multimodal models [13].

## References
[1] Video Diffusion Models. arxiv. https://arxiv.org/abs/2204.03458 (2022-04-06)
[2] A Survey on Video Diffusion Models. arxiv. https://arxiv.org/abs/2310.10647 (2023-10-16)
[3] Open-Sora: Democratizing Efficient Video Production for All. hf-search. https://huggingface.co/papers/2412.20404 (2024-12-29)
[4] HunyuanVideo: A Systematic Framework For Large Video Generative Models. hf-search. https://huggingface.co/papers/2412.03603 (2024-12-03)
[5] Make-A-Video: Text-to-Video Generation without Text-Video Data. arxiv. https://arxiv.org/abs/2209.14792 (2022-09-29)
[6] Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. hf-search. https://huggingface.co/papers/2311.15127 (2023-11-25)
[7] AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. hf-search. https://huggingface.co/papers/2307.04725 (2023-07-12)
[8] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. hf-search. https://huggingface.co/papers/2408.06072 (2024-08-12)
[9] Sora Generates Videos with Stunning Geometrical Consistency. hf-search. https://huggingface.co/papers/2402.17403 (2024-02-27)
[10] VBench: Comprehensive Benchmark Suite for Video Generative Models. arxiv. https://arxiv.org/abs/2311.17982 (2023-11-29)
[11] EvalCrafter: Benchmarking and Evaluating Large Video Generation Models. web. https://evalcrafter.github.io/ (2023-10-18)
[12] Adaptive Caching for Faster Video Generation with Diffusion Transformers. web. https://arxiv.org/html/2411.02397v2 (2024-11-04)
[13] Safe-Sora & VideoShield: Copyright, Watermarking, and Safety Alignment. web. https://proceedings.neurips.cc/paper_files/paper/e5eaf67f3405be58cd12848a89cd8ace-Paper-Conference.pdf (2025-05-18)
[14] Imagen Video: High Definition Video Generation with Diffusion Models. arxiv. https://arxiv.org/abs/2210.02303 (2022-10-05)
[15] Compact Attention: Exploiting Structured Spatio-Temporal Sparsity for Fast Video Generation. hf-search. https://huggingface.co/papers/2508.12969 (2025-08-18)
[16] HunyuanVideo 1.5 Technical Report. hf-search. https://huggingface.co/papers/2511.18870 (2025-11-24)
[17] How Far is Video Generation from World Model: A Physical Law Perspective. hf-search. https://huggingface.co/papers/2411.02385 (2024-11-04)
[18] Fast and Memory-Efficient Video Diffusion Using Streamlined Inference. web. https://proceedings.neurips.cc/paper_files/paper/2024/file/18b0b4c788c8f2cf6c2943b989ad18c8-Paper-Conference.pdf (2024-12-10)
