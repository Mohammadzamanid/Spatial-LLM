# Draft cover letter — Nature Machine Intelligence

Dear Editors,

Please consider our Article, **“Causal cognitive-map interfaces for language models: when brain-inspired spatial codes help, and when they do not,”** for publication in *Nature Machine Intelligence*.

Large language models can express spatial knowledge, but this does not establish that they maintain a metric state that is updated through self-motion. We test a complementary route: a structured cognitive-map interface that path-integrates self-motion into a bounded multi-scale population code and exposes that latent state to a frozen language model through gated cross-attention.

The manuscript makes four linked contributions. First, causal cortex-ON/OFF and organ-specific lesion experiments show that spatial answers are carried through the latent spatial channel rather than recovered from the text prompt. Second, matched non-neural baselines identify an important boundary condition: ordinary Euclidean path integration can be matched by a simpler additive Transformer, whereas periodicity, remapping and population structure become useful only in regimes where those properties are computationally load-bearing. Third, a prospective theta-sweep signal improves blocked-ahead reasoning and independently replicates across Qwen2.5-1.5B and SmolLM2-1.7B. Fourth, the fixed representation transfers without architectural changes to user-disjoint real human trajectories from Microsoft GeoLife, where it supports accurate bearing and distance decoding across unseen users.

We believe the work is a strong fit for *Nature Machine Intelligence* because it connects mechanistic neuroscience-inspired representation learning with causal analysis of language-model behaviour, while explicitly testing when biological inductive biases do and do not provide computational value. The paper is not framed as a universal claim that grid codes outperform conventional machine-learning representations; its contribution is a falsifiable regime map of when structured cognitive maps change model behaviour, supported by null results, ablations, cross-backbone replication and external validation.

All central quantitative claims are linked to committed result artifacts and reproducible scripts. The exact submission code will be archived with a persistent DOI. Extended neuroscience-inspired analyses that are not necessary for the central claim have been moved to Supplementary Information to keep the main Article focused.

[AUTHOR CONFIRMATION REQUIRED: This manuscript is not under consideration elsewhere and all authors have approved the submission.]

[AUTHOR CONFIRMATION REQUIRED: State whether there are any competing interests or closely related manuscripts/preprints that the editors should know about.]

Thank you for considering our manuscript.

Sincerely,

Seyyed Mohammad Ali Zamani Deravei  
Corresponding author  
[PRIMARY AFFILIATION]  
[EMAIL]
