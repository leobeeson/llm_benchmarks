# llm_benchmarks
A collection of benchmarks and datasets for evaluating LLM.

## Knowledge and Language Understanding

### Massive Multitask Language Understanding (MMLU)

* **Description:** Measures general knowledge across 57 different subjects, ranging from STEM to social sciences.
* **Purpose:** To assess the LLM's understanding and reasoning in a wide range of subject areas.
* **Relevance:** Ideal for multifaceted AI systems that require extensive world knowledge and problem solving ability.
* **Source:** [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
* **Resources:**
  * [MMLU GitHub](https://github.com/hendrycks/test)
  * [MMLU Dataset](https://people.eecs.berkeley.edu/~hendrycks/data.tar)

### AI2 Reasoning Challenge (ARC)

* **Description:** Tests LLMs on grade-school science questions, requiring both deep general knowledge and reasoning abilities.
* **Purpose:** To evaluate the ability to answer complex science questions that require logical reasoning.
* **Relevance:** Useful for educational AI applications, automated tutoring systems, and general knowledge assessments.
* **Source:** [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457)
* **Resources:**
  * [ARC Dataset: HuggingFace](https://huggingface.co/datasets/ai2_arc)
  * [ARC Dataset: Allen Institute](https://allenai.org/data/arc)

### General Language Understanding Evaluation (GLUE)

* **Description:** A collection of various language tasks from multiple datasets, designed to measure overall language understanding.
* **Purpose:** To provide a comprehensive assessment of language understanding abilities in different contexts.
* **Relevance:** Crucial for applications requiring advanced language processing, such as chatbots and content analysis.
* **Source:** [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
* **Resources:**
  * [GLUE Homepage](https://gluebenchmark.com/)
  * [GLUE Dataset](https://huggingface.co/datasets/glue)

### Natural Questions

* **Description:** A collection of real-world questions people have Googled, paired with relevant Wikipedia pages to extract answers.
* **Purpose:** To test the ability to find accurate short and long answers from web-based sources.
* **Relevance:** Essential for search engines, information retrieval systems, and AI-driven question-answering tools.
* **Source:** [Natural Questions: A Benchmark for Question Answering Research](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518/Natural-Questions-A-Benchmark-for-Question)
* **Resources:**
  * [Natural Questions Homepage](https://ai.google.com/research/NaturalQuestions)
  * [Natural Questions Dataset: Github](https://github.com/google-research-datasets/natural-questions)

### LAnguage Modelling Broadened to Account for Discourse Aspects (LAMBADA)

* **Description:** A collection of passages testing the ability of language models to understand and predict text based on long-range context.
* **Purpose:** To assess the models' comprehension of narratives and their predictive abilities in text generation.
* **Relevance:** Important for AI applications in narrative analysis, content creation, and long-form text understanding.
* **Source:** [The LAMBADA Dataset: Word prediction requiring a broad discourse context](https://arxiv.org/abs/1606.06031)
* **Resources:**
  * [LAMBADA Dataset: HuggingFace](https://huggingface.co/datasets/lambada)

### HellaSwag

* **Description:** Tests natural language inference by requiring LLMs to complete passages in a way that requires understanding intricate details.
* **Purpose:** To evaluate the model's ability to generate contextually appropriate text continuations.
* **Relevance:** Useful in content creation, dialogue systems, and applications requiring advanced text generation capabilities.
* **Source:** [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)
* **Resources:**
  * [HellaSwag Dataset: GitHub](https://github.com/rowanz/hellaswag/tree/master/data)

### Multi-Genre Natural Language Inference (MultiNLI)

* **Description:** A benchmark consisting of 433K sentence pairs across various genres of English data, testing natural language inference.
* **Purpose:** To assess the ability of LLMs to assign correct labels to hypothesis statements based on premises.
* **Relevance:** Vital for systems requiring advanced text comprehension and inference, such as automated reasoning and text analytics tools.
* **Source:** [A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426)
* **Resources:**
  * [MultiNLI Homepage](https://cims.nyu.edu/~sbowman/multinli/)
  * [MultiNLI Dataset](https://huggingface.co/datasets/multi_nli)

### SuperGLUE

* **Description:** An advanced version of the GLUE benchmark, comprising more challenging and diverse language tasks.
* **Purpose:** To evaluate deeper aspects of language understanding and reasoning.
* **Relevance:** Important for sophisticated AI systems requiring advanced language processing capabilities.
* **Source:** [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://arxiv.org/abs/1905.00537)
* **Resources:**
  * [SuperGLUE Dataset: HuggingFace](https://huggingface.co/datasets/super_glue)

### TriviaQA

* **Description:** A reading comprehension test with questions from sources like Wikipedia, demanding contextual analysis.
* **Purpose:** To assess the ability to sift through context and find accurate answers in complex texts.
* **Relevance:** Suitable for AI systems in knowledge extraction, research, and detailed content analysis.
* **Source:** [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://arxiv.org/abs/1705.03551)
* **Resources:**
  * [TriviaQA GitHub](https://github.com/mandarjoshi90/triviaqa)
  * [TriviaQa Dataset](https://huggingface.co/datasets/trivia_qa)

### WinoGrande

* **Description:** A large set of problems based on the Winograd Schema Challenge, testing context understanding in sentences.
* **Purpose:** To evaluate the ability of LLMs to grasp nuanced context and subtle variations in text.
* **Relevance:** Crucial for models dealing with narrative analysis, content personalization, and advanced text interpretation.
* **Source:** [WinoGrande: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/abs/1907.10641)
* **Resources:**
  * [WinoGrande GitHub](https://github.com/allenai/winogrande)
  * [WinoGrande Dataset: HuggingFace](https://huggingface.co/datasets/winogrande)

### SciQ

* **Description:** Consists of multiple-choice questions mainly in natural sciences like physics, chemistry, and biology.
* **Purpose:** To test the ability to answer science-based questions, often with additional supporting text.
* **Relevance:** Useful for educational tools, especially in science education and knowledge testing platforms.
* **Source:** [Crowdsourcing Multiple Choice Science Questions](https://arxiv.org/abs/1707.06209)
* **Resources:**
  * [SciQ Dataset: HuggingFace](https://huggingface.co/datasets/sciq)

## Reasoning Capabilities

### GSM8K

* **Description:** A set of 8.5K grade-school math problems that require basic to intermediate math operations.
* **Purpose:** To test LLMs’ ability to work through multistep math problems.
* **Relevance:** Useful for assessing AI’s capability in solving basic mathematical problems, valuable in educational contexts.
* **Source:** [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
* **Resources:**
  * [GSM8K Dataset](https://huggingface.co/datasets/gsm8k)

### Discrete Reasoning Over Paragraphs (DROP)

* **Description:** An adversarially-created reading comprehension benchmark requiring models to navigate through references and execute operations like addition or sorting.
* **Purpose:** To evaluate the ability of models to understand complex texts and perform discrete operations.
* **Relevance:** Useful in advanced educational tools and text analysis systems requiring logical reasoning.
* **Source**: [DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://arxiv.org/abs/1903.00161)
* **Resources:**:
  * [DROP Dataset](https://huggingface.co/datasets/drop)

### Counterfactual Reasoning Assessment (CRASS)

* **Description:** Evaluates counterfactual reasoning abilities of LLMs, focusing on "what if" scenarios.
* **Purpose:** To assess models' ability to understand and reason about alternate scenarios based on given data.
* **Relevance:** Important for AI applications in strategic planning, decision-making, and scenario analysis.
* **Source**: [CRASS: A Novel Data Set and Benchmark to Test Counterfactual Reasoning of Large Language Models](https://arxiv.org/abs/2112.11941)
* **Resources:**
  * [CRASS Dataset](https://github.com/apergo-ai/CRASS-data-set/tree/main)

### Large-scale ReAding Comprehension Dataset From Examinations (RACE)

* **Description:** A set of reading comprehension questions derived from English exams given to Chinese students.
* **Purpose:** To test LLMs' understanding of complex reading material and their ability to answer examination-level questions.
* **Relevance:** Useful in language learning applications and educational systems for exam preparation.
* **Source:** [RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://arxiv.org/abs/1704.04683)
* **Resources:**
  * [RAC Dataset](https://www.cs.cmu.edu/~glai1/data/race/)

### Big-Bench Hard (BBH)

* **Description:** A subset of BIG-Bench focusing on the most challenging tasks requiring multi-step reasoning.
* **Purpose:** To challenge LLMs with complex tasks demanding advanced reasoning skills.
* **Relevance:** Important for evaluating the upper limits of AI capabilities in complex reasoning and problem-solving.
* **Source:** [Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them](https://arxiv.org/abs/2210.09261)
* **Resources:**
  * [BIG-Bench-Hard GitHub: Dataset and Prompts](https://github.com/suzgunmirac/BIG-Bench-Hard)
  * [BBH Dataset: HuggingFace](https://huggingface.co/datasets/lukaemon/bbh)

### AGIEval

* **Description:** A collection of standardized tests, including GRE, GMAT, SAT, LSAT, and civil service exams.
* **Purpose:** To evaluate LLMs' reasoning abilities and problem-solving skills across various academic and professional scenarios.
* **Relevance:** Useful for assessing AI capabilities in standardized testing and professional qualification contexts.
* **Source:** [AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models](https://arxiv.org/abs/2304.06364)
* **Resources:**
  * [AGIEval Github: Dataset and Prompts](https://github.com/ruixiangcui/AGIEval/tree/main)
  * [AGIEval Datasets: HuggingFace](https://huggingface.co/datasets?search=AGIEval)

### BoolQ

* **Description:** A collection of over 15,000 real yes/no questions from Google searches, paired with Wikipedia passages.
* **Purpose:** To test the ability of LLMs to infer correct answers from contextual information that may not be explicit.
* **Relevance:** Crucial for question-answering systems and knowledge-based AI applications where accurate inference is key.
* **Source:** [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://arxiv.org/abs/1905.10044)
* **Resources:**
  * [BoolQ Dataset: HuggingFace](https://huggingface.co/datasets/boolq)

## Multi Turn Open Ended Conversations

### MT-bench

* **Description:** Tailored for evaluating the proficiency of chat assistants in sustaining multi-turn conversations.
* **Purpose:** To test the ability of models to engage in coherent and contextually relevant dialogues over multiple turns.
* **Relevance:** Essential for developing sophisticated conversational agents and chatbots.
* **Source:** [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
* **Resources:**
  * [MT-bench Human Annotation Dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)

### Question Answering in Context (QuAC)

* **Description:** Features 14,000 dialogues with 100,000 question-answer pairs, simulating student-teacher interactions.
* **Purpose:** To challenge LLMs with context-dependent, sometimes unanswerable questions within dialogues.
* **Relevance:** Useful for conversational AI, educational software, and context-aware information systems.
* **Source:** [QuAC : Question Answering in Context](https://arxiv.org/abs/1808.07036)
* **Resources:**
  * [QuAC Homepage and Dataset](https://quac.ai/)

## Grounding and Abstractive Summarization

### Ambient Clinical Intelligence Benchmark (ACI-BENCH)

* **Description:** Contains full doctor-patient conversations and associated clinical notes from various medical domains.
* **Purpose:** To challenge models to accurately generate clinical notes based on conversational data.
* **Relevance:** Vital for AI applications in healthcare, especially in automated documentation and medical analysis.
* **Source:**: [ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation](https://arxiv.org/abs/2306.02022)
* **Resources:**:
  * [ACI-BENCH Dataset](https://github.com/wyim/aci-bench)

### MAchine Reading COmprehension Dataset (MS-MARCO)

* **Description:** A large-scale collection of natural language questions and answers derived from real web queries.
* **Purpose:** To test the ability of models to accurately understand and respond to real-world queries.
* **Relevance:** Crucial for search engines, question-answering systems, and other consumer-facing AI applications.
* **Source:** [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268)
* **Resources:**
  * [MS-MARCO Dataset](https://huggingface.co/datasets/ms_marco)

### Query-based Multi-domain Meeting Summarization (QMSum)

* **Description:** A benchmark for summarizing relevant spans of meetings in response to specific queries.
* **Purpose:** To evaluate the ability of models to extract and summarize important information from meeting content.
* **Relevance:** Useful for business intelligence tools, meeting analysis applications, and automated summarization systems.
* **Source:** [QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization](https://arxiv.org/abs/2104.05938)
* **Resources:**
  * [QMSum Dataset](https://github.com/Yale-LILY/QMSum)

### Physical Interaction: Question Answering (PIQA)

* **Description:** Tests knowledge and understanding of the physical world through hypothetical scenarios and solutions.
* **Purpose:** To measure the model’s capability in handling physical interaction scenarios.
* **Relevance:** Important for AI applications in robotics, physical simulations, and practical problem-solving systems.
* **Source:**: [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/abs/1911.11641)
* **Resources:**
  * [PIQA Dataset: GitHub](https://github.com/ybisk/ybisk.github.io/tree/master/piqa)

## Content Moderation and Narrative Control

### ToxiGen

* **Description:** A dataset of toxic and benign statements about minority groups, focusing on implicit hate speech.
* **Purpose:** To test a model's ability to both identify and avoid generating toxic content.
* **Relevance:** Crucial for content moderation systems, community management, and AI ethics research.
* **Source:** [ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection](https://arxiv.org/abs/2203.09509)
* **Resources:**
  * [TOXIGEN Code and Prompts: GitHub](https://github.com/microsoft/TOXIGEN/tree/main)
  * [TOXIGEN Dataset: HuggingFace](https://huggingface.co/datasets/skg/toxigen-data)

### Helpfulness, Honesty, Harmlessness (HHH)

* **Description:** Evaluates language models' alignment with ethical standards such as helpfulness, honesty, and harmlessness.
* **Purpose:** To assess the ethical responses of models in interaction scenarios.
* **Relevance:** Vital for ensuring AI systems promote positive interactions and adhere to ethical standards.
* **Source:** [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861)
* **Resources:**
  * [HH-RLHF Datasets: GitHub](https://github.com/anthropics/hh-rlhf)
  * Related Research:
    * [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
    * [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858)

### TruthfulQA

* **Description:** A benchmark for evaluating the truthfulness of LLMs in generating answers to questions prone to false beliefs and biases.
* **Purpose:** To test the ability of models to provide accurate and unbiased information.
* **Relevance:** Important for AI systems where delivering accurate and unbiased information is critical, such as in educational or advisory roles.
* **Source:** [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958v2)
* **Resources:**
  * [TruthfulQA Dataset: GitHub](https://github.com/sylinrl/TruthfulQA)

### Responsible AI (RAI)

* **Description:** A framework for evaluating the safety of chat-optimized models in conversational settings.
* **Purpose:** To assess potential harmful content, IP leakage, and security breaches in AI-driven conversations.
* **Relevance:** Crucial for developing safe and secure conversational AI applications, particularly in sensitive domains.
* **Source:** [A Framework for Automated Measurement of Responsible AI Harms in Generative AI Applications](https://arxiv.org/abs/2310.17750)

## Coding Capabilities

### CodeXGLUE

* **Description:** Evaluates LLMs' ability to understand and work with code across various tasks like code completion and translation.
* **Purpose:** To assess code intelligence, including understanding, fixing, and explaining code.
* **Relevance:** Essential for applications in software development, code analysis, and technical documentation.
* **Source:** [CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation](https://arxiv.org/abs/2102.04664)
* **Reference:**
  * [CodeXGLUE Dataset: GitHub](https://github.com/microsoft/CodeXGLUE)

### HumanEval

* **Description:** Contains programming challenges for evaluating LLMs' ability to write functional code based on instructions.
* **Purpose:** To test the generation of correct and efficient code from given requirements.
* **Relevance:** Important for automated code generation tools, programming assistants, and coding education platforms.
* **Source:** [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
* **Resources:**
  * [HumanEval Dataset: GitHub](https://github.com/openai/human-eval)

### Mostly Basic Python Programming (MBPP)

* **Description:** Includes 1,000 Python programming problems suitable for entry-level programmers.
* **Purpose:** To evaluate proficiency in solving basic programming tasks and understanding of Python.
* **Relevance:** Useful for beginner-level coding education, automated code generation, and entry-level programming testing.
* **Source:** [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732)
* **Resources:**
  * [MBPP Dataset: HuggingFace](https://huggingface.co/datasets/mbpp)

## LLM-Assisted Evaluation

### LLM Judge

* **Source:** [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
* **Abstract:** Evaluating large language model (LLM) based chat assistants is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences. To address this, we explore using strong LLMs as judges to evaluate these models on more open-ended questions. We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate some of them. We then verify the agreement between LLM judges and human preferences by introducing two benchmarks: MT-bench, a multi-turn question set; and Chatbot Arena, a crowdsourced battle platform. Our results reveal that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement between humans. Hence, LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain. Additionally, we show our benchmark and traditional benchmarks complement each other by evaluating several variants of LLaMA and Vicuna. The MT-bench questions, 3K expert votes, and 30K conversations with human preferences are publicly available at this https URL.
* **Insights:**
  * Use MT-bench questions and prompts to evaluate your models with LLM-as-a-judge. MT-bench is a set of challenging multi-turn open-ended questions for evaluating chat assistants. To automate the evaluation process, we prompt strong LLMs like GPT-4 to act as judges and assess the quality of the models' responses.
* **Resources:**
  * [LLM Judge GitHub](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
  * [MT-bench and Arena Elo Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
  * [Chatbot Arena](https://chat.lmsys.org/?arena)
  * LLM Judge Datasets:
    * [LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
    * [Chatbot Arena Conversation Dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
    * [MT-bench Human Annotation Dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)

### LLM-Eval

* **Source:** [Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models](https://arxiv.org/abs/2305.13711)
* **Abstract:** We propose LLM-Eval, a unified multi-dimensional automatic evaluation method for open-domain conversations with large language models (LLMs). Existing evaluation methods often rely on human annotations, ground-truth responses, or multiple LLM prompts, which can be expensive and time-consuming. To address these issues, we design a single prompt-based evaluation method that leverages a unified evaluation schema to cover multiple dimensions of conversation quality in a single model call. We extensively evaluate the performance of LLM-Eval on various benchmark datasets, demonstrating its effectiveness, efficiency, and adaptability compared to state-of-the-art evaluation methods. Our analysis also highlights the importance of choosing suitable LLMs and decoding strategies for accurate evaluation results. LLM-Eval offers a versatile and robust solution for evaluating open-domain conversation systems, streamlining the evaluation process and providing consistent performance across diverse scenarios.
* **Insights:**
  * Top-shelve LLM (e.g. GPT4, Claude) correlate better with human score than metric-based eval measures.

### JudgeLM

* **Source:** [JudgeLM: Fine-tuned Large Language Models are Scalable Judges](https://arxiv.org/abs/2310.17631)
* **Abstract:** Evaluating Large Language Models (LLMs) in open-ended scenarios is challenging because existing benchmarks and metrics can not measure them comprehensively. To address this problem, we propose to fine-tune LLMs as scalable judges (JudgeLM) to evaluate LLMs efficiently and effectively in open-ended benchmarks. We first propose a comprehensive, large-scale, high-quality dataset containing task seeds, LLMs-generated answers, and GPT-4-generated judgments for fine-tuning high-performance judges, as well as a new benchmark for evaluating the judges. We train JudgeLM at different scales from 7B, 13B, to 33B parameters, and conduct a systematic analysis of its capabilities and behaviors. We then analyze the key biases in fine-tuning LLM as a judge and consider them as position bias, knowledge bias, and format bias. To address these issues, JudgeLM introduces a bag of techniques including swap augmentation, reference support, and reference drop, which clearly enhance the judge's performance. JudgeLM obtains the state-of-the-art judge performance on both the existing PandaLM benchmark and our proposed new benchmark. Our JudgeLM is efficient and the JudgeLM-7B only needs 3 minutes to judge 5K samples with 8 A100 GPUs. JudgeLM obtains high agreement with the teacher judge, achieving an agreement exceeding 90% that even surpasses human-to-human agreement. JudgeLM also demonstrates extended capabilities in being judges of the single answer, multimodal models, multiple answers, and multi-turn chat.
* **Insights:**
  * Relatively small models (e.g 7b models) can be fine-tuned to be reliable judges of other models.

### Prometheus

* **Source:** [Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491)
* **Abstract:** Recently, using a powerful proprietary Large Language Model (LLM) (e.g., GPT-4) as an evaluator for long-form responses has become the de facto standard. However, for practitioners with large-scale evaluation tasks and custom criteria in consideration (e.g., child-readability), using proprietary LLMs as an evaluator is unreliable due to the closed-source nature, uncontrolled versioning, and prohibitive costs. In this work, we propose Prometheus, a fully open-source LLM that is on par with GPT-4's evaluation capabilities when the appropriate reference materials (reference answer, score rubric) are accompanied. We first construct the Feedback Collection, a new dataset that consists of 1K fine-grained score rubrics, 20K instructions, and 100K responses and language feedback generated by GPT-4. Using the Feedback Collection, we train Prometheus, a 13B evaluator LLM that can assess any given long-form text based on customized score rubric provided by the user. Experimental results show that Prometheus scores a Pearson correlation of 0.897 with human evaluators when evaluating with 45 customized score rubrics, which is on par with GPT-4 (0.882), and greatly outperforms ChatGPT (0.392). Furthermore, measuring correlation with GPT-4 with 1222 customized score rubrics across four benchmarks (MT Bench, Vicuna Bench, Feedback Bench, Flask Eval) shows similar trends, bolstering Prometheus's capability as an evaluator LLM. Lastly, Prometheus achieves the highest accuracy on two human preference benchmarks (HHH Alignment & MT Bench Human Judgment) compared to open-sourced reward models explicitly trained on human preference datasets, highlighting its potential as an universal reward model.
* **Insights:**
  * A scoring  rubric and a reference answer vastly improve correlation with human scores.
 
## Industry Resources

* [Latent Space - Benchmarks 201: Why Leaderboards > Arenas >> LLM-as-Judge](https://www.latent.space/p/benchmarks-201?publication_id=1084089&post_id=146497374&isFreemail=true&r=53qrp&triedRedirect=true)
    *  **Summary:**
        * The OpenLLM Leaderboard, maintained by Clémentine Fourrier, is a standardized and reproducible way to evaluate language models' performance.
        * The leaderboard initially gained popularity in summer 2023 and has had over 2 million unique visitors and 300,000 active community members.
        * The recent update to the leaderboard (v2) includes six benchmarks to address model overfitting and to provide more room for improved performance.
        * LLMs are not recommended as judges due to issues like mode collapse and positional bias.
        * If LLMs must be used as judges, open LLMs like Prometheus or JudgeLM are suggested for reproducibility.
        * The LMSys Arena is another platform for AI engineers, but its rankings are not reproducible and may not accurately reflect model capabilities.
