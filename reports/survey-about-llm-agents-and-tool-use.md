# A Comprehensive Survey on Large Language Model Agents and Tool Use

## TL;DR
- **Paradigm Shift from Generation to Agency**: Large Language Model (LLM) agents combine frozen or fine-tuned language models with external tool use (such as calculators, search engines, APIs, and operating systems) to operate autonomously in interactive, partially observable environments [1][2][3].
- **Core Reasoning & Planning Architectures**: Breakthrough frameworks like ReAct, Toolformer, and Chameleon demonstrate that interleaving verbal reasoning traces with tool execution dramatically reduces hallucination and improves multi-step decision making [1][4][5].
- **Recent Frontier Advancements**: Modern agentic research has expanded toward complex execution environments, including real-world desktop environments (OSWorld), software repositories (SWE-bench), and multi-agent collaborative frameworks governed by Standardized Operating Procedures (MetaGPT) [6][7][8].
- **Evaluation Challenges and Enterprise Trade-offs**: Comprehensive benchmarks (AgentBench, AgentProp-Bench) reveal significant performance gaps between top commercial models and open-source alternatives, while exposing critical failure modes including tool-call hallucination, error propagation cascades, and high latency/cost in enterprise deployment [2][9][10].

## Background
The rapid evolution of Large Language Models (LLMs) has catalyzed a paradigm shift from passive text generation to active agency. Early language models operated primarily in static input-output settings, limited by parametric knowledge cutoffs, inability to interact with real-time data, and vulnerability to factual hallucination. To overcome these constraints, researchers introduced agentic architectures that endow LLMs with tool-use capabilities.

Foundational breakthroughs established that treating language models as reasoning engines capable of invoking external tools transforms their capabilities. For instance, **ReAct** (Reasoning and Acting) established a paradigm prompting LLMs to generate both verbal reasoning traces and task-specific actions in an interleaved manner [1]. By interleaving reasoning and action, models can dynamically update plans, handle unexpected observations, and query external APIs or knowledge bases (e.g., Wikipedia), overcoming static error propagation [1]. Similarly, **Toolformer** demonstrated that language models can be trained in a self-supervised manner to decide which APIs to call, what arguments to pass, and how to incorporate tool outputs into token generation [4]. Concurrently, **WebGPT** showed that human feedback and behavior cloning enable models to navigate text-based browsers and cite references effectively for question answering [11], while **Chameleon** introduced plug-and-play compositional reasoning by synthesizing programs from diverse tools such as vision models and Python interpreters [5].

## Core Methods and Taxonomies of Tool Use
Agentic tool use is formally modeled as decision-making within Partially Observable Markov Decision Processes (POMDPs), where an agent navigates a state space $S$, action space $A$, transition function $T$, reward function $R$, and observation space $O$ [2]. To navigate these environments successfully, modern frameworks rely on structured taxonomies and methodological modules:

1. **Planning and Decomposition:** Complex tasks are decomposed into manageable sub-tasks using hierarchical reasoning structures such as Tree of Thoughts, graph-based planners, or explicit plan-and-solve prompts [12].
2. **API Interaction and Function Calling:** Rather than relying solely on raw text generation, models leverage structured function-calling APIs that enforce strict JSON schemas for arguments, parameters, and return types [12][10].
3. **Training-Free Experience Reuse and Dynamic Creation:** Recent advancements explore training-free frameworks where agents dynamically create, refine, and reuse custom tools or execution snippets during inference, shifting from static tool users to active tool creators [13].
4. **Self-Correction and Reflection:** Frameworks like Reflexion enable agents to evaluate intermediate execution outputs, ingest runtime feedback, and iteratively correct faulty tool calls or flawed reasoning paths [12].

Despite these methodological advancements, systematic audits of evaluation benchmarks highlight considerable evaluator misalignment and metric variance, emphasizing the need for robust execution-based validation rather than superficial substring matching [14].

## Frontier Advancements in Multimodal and Multi-Agent Systems
As LLMs evolve into frontier multimodal and multi-agent systems, agentic tool use has expanded into complex, open-ended environments requiring rich sensory grounding and collaborative division of labor:

- **Multimodal OS & Desktop Agents:** Environments such as **OSWorld** provide execution-based evaluation across real operating systems (Ubuntu, Windows, macOS) and desktop applications (Chrome, LibreOffice, GIMP) [6]. While human operators achieve high success rates (72.36%), state-of-the-art multimodal models (GPT-4V, Gemini, Claude-3 Opus) achieve under 12.24% success [6]. Primary bottlenecks include GUI grounding, precise coordinate prediction from screenshots, handling background noise/popups, and maintaining operational state across multi-app workflows [6].
- **Software Engineering Agents:** **SWE-bench** established a rigorous benchmark for resolving real-world GitHub issues by requiring models to navigate, edit, and test multi-file codebases [7]. This benchmark has catalyzed a thriving ecosystem of specialized environments, including SWE-bench Multimodal for visual JavaScript issues and SWE-bench-Live for continuous contamination-resistant evaluation [7].
- **Multi-Agent Collaborative Frameworks:** To mitigate cascading hallucinations and context drift inherent in free-form multi-agent chat, frameworks like **MetaGPT** encode Standardized Operating Procedures (SOPs) into prompt sequences [8]. By enforcing structured document exchange (Product Requirement Documents, class diagrams, sequence flows) and a shared global message pool with publish-subscribe mechanisms, MetaGPT achieves state-of-the-art performance on complex software engineering tasks [8]. Furthermore, data-centric self-adaptive frameworks ("Learn-by-interact") synthesize robust interaction data to improve agent performance across coding and browsing domains [15].

## Applications, Benchmarks, and Enterprise Trade-offs
Evaluating LLM agents requires multi-dimensional frameworks that measure behavior, capabilities, reliability, and safety [10]. Prominent benchmarks such as **AgentBench** evaluate LLMs-as-Agents across diverse interactive environments (Operating Systems, Databases, Knowledge Graphs, Games, Web shopping) [2][3]. These evaluations reveal pronounced performance gaps between top commercial models and open-source models under 70B parameters, while identifying recurrent failure modes such as Context Limit Exceeded (CLE), Invalid Format (IF), Invalid Action (IA), and Task Limit Exceeded (TLE) [2][3].

Recent empirical studies, such as **AgentProp-Bench**, demonstrate that parameter-level tool errors propagate to final task failures with a human-calibrated probability of $\approx 0.62$, and that an agent's ability to reject corrupted inputs is statistically independent from its ability to recover from them ($\rho = 0.041$) [9]. Furthermore, models frequently exhibit "fabricated tool use"—asserting tool-derived results never actually obtained in up to 37.5% of traces [9].

In enterprise deployments, these dynamics introduce severe trade-offs involving reliability, latency, and cost [10]. Multi-turn tool loops significantly increase End-to-End Request Latency and operational expenses. Consequently, production architectures increasingly integrate dual-axis robustness testing, runtime interceptors, observability platforms (Langfuse, AgentOps, OpenLLMetry), and security guardrails (NeMo Guardrails, Guardrails AI) to govern agent trajectories and mitigate prompt injection and data leaks [9][16].

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[2] AgentBench: Evaluating LLMs as Agents. arxiv. https://arxiv.org/abs/2308.03688 (2025-10-04)
[3] AgentBench: Evaluating LLMs as Agents. web. https://proceedings.iclr.cc/paper_files/paper/2024/file/e9df36b21ff4ee211a8b71ee8b7e9f57-Paper-Conference.pdf (2024-05-01)
[4] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[5] Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models. arxiv. https://proceedings.neurips.cc/paper_files/paper/2023/hash/871ed095b734818cfba48db6aeb25a62-Abstract.html (2023-12-15)
[6] OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. hf-search. https://huggingface.co/papers/2404.07972 (2024-04-11)
[7] SWE-bench: Can Language Models Resolve Real-World GitHub Issues?. hf-search. https://huggingface.co/papers/2310.06770 (2023-10-10)
[8] MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework. web. https://arxiv.org/abs/2308.00352 (2023-08-01)
[9] Evaluating Tool-Using Language Agents: Judge Reliability, Propagation Cascades, and Runtime Mitigation in AgentProp-Bench. web. https://arxiv.org/html/2604.16706 (2026-04-16)
[10] Evaluation and Benchmarking of LLM Agents: A Survey. web. https://arxiv.org/html/2507.21504v1 (2025-07-21)
[11] WebGPT: Browser-assisted question-answering with human feedback. arxiv. https://arxiv.org/abs/2112.09332 (2021-12-17)
[12] Agentic Reasoning for Large Language Models. hf-search. https://huggingface.co/papers/2601.12538 (2026-01-18)
[13] Evolving from Tool User to Creator via Training-Free Experience Reuse in Multimodal Reasoning. hf-search. https://huggingface.co/papers/2602.01983 (2026-02-02)
[14] Benchmarking the Benchmarks: A Validity Audit of Tool-Calling Evaluation. hf-search. https://huggingface.co/papers/2607.02577 (2026-06-30)
[15] Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments. hf-search. https://huggingface.co/papers/2501.10893 (2025-01-18)
[16] Awesome Agent Evals & Enterprise Benchmarks. web. https://github.com/royalpinto007/awesome-agent-evals/blob/main/README.md (2025-01-01)
