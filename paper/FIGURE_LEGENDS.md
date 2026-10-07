# Main figure legends

## Figure 1 | Causal language transfer through a structured spatial channel

**a,** System schematic. Self-motion is integrated by the cognitive-map substrate and injected into a frozen language model through gated spatial-to-text fusion; movement information is not included in the language prompt.  
**b,** Leakage-controlled toroidal spatial readout in Qwen2.5-1.5B. Cortex-ON exact accuracy is shown across path lengths 8, 16 and 24 against the text-only cortex-OFF control; bars show means and 95% confidence intervals across six seeds.  
**c,** Representative causal language readouts for elapsed time, allocentric position (WHERE), head direction (FACING) and egocentric landmark direction (LANDMARK).  
**d,** Organ-specific lesion tests. For each task, the intact system is compared with lesioning the task-relevant spatial organ and with lesioning a different organ. Selective impairment establishes a double dissociation between latent subcodes and language outputs. Source data: `results/torus_llm.json`, `results/elapsed_time_llm.json`, `results/deadreckoning_llm_agg.json`, `results/multiframe_llm_agg.json`.

## Figure 2 | The value of grid-like coding is regime-dependent

**a,** On ordinary Euclidean path integration at T=24, the constrained grid code and a permutation-invariant NoPE+sum Transformer are statistically indistinguishable (paired permutation p=0.94; Cohen's d=0.04), establishing a certified null.  
**b,** The bounded place-cell baseline loses range under extrapolation beyond the training region, whereas the grid code retains higher distance accuracy.  
**c,** Summary regime map across Euclidean extrapolation, cyclic worlds, one-shot capacity, context-free versus labelled multi-map settings, low-data learning and integration noise. Values are the committed task metrics; text annotations indicate win, tie or loss according to the experiment-specific comparison. Source data: `results/significance.json`, `results/phase_diagram.json`, `results/extrapolation.json`.

## Figure 3 | Periodicity and remapping become useful only when their structure is load-bearing

**a,** Toroidal path integration. The periodic grid representation remains at ceiling as path length and wrap count increase, whereas non-periodic NoPE+sum and Euclidean place representations collapse toward chance. Error bars show 95% confidence intervals across eight seeds.  
**b,** Context-free one-shot multi-map memory. Environment-specific remapping prevents collisions as the number of contexts grows; turning remapping off or using a deterministic raw displacement code causes rapid failure.  
**c,** Labelled multi-map control. When an explicit context identity is provided to a trained model, non-remapping representations remain accurate, showing that an external context signal can substitute for internal remapping. Source data: `results/torus.json`, `results/code_necessity.json`, `results/multimap_task.json`.

## Figure 4 | Prospective theta-sweep information is causally load-bearing and replicates across LLM families

**a,** Qwen2.5-1.5B blocked-ahead task in novel layouts. Cortex-ON is compared with cortex-OFF, sweep ablation (NO-SWEEP) and wrong-heading sweep controls. Bars show mean accuracy and 95% confidence intervals across eight seeds; dashed line denotes binary chance.  
**b,** Independent replication in SmolLM2-1.7B-Instruct using forced-choice 0/1 scoring.  
**c,** Per-seed SmolLM2 ON−NO-SWEEP effects. Seven of eight seeds are positive and one is tied; exact paired sign-flip p=0.0156. Source data: `results/theta_sweep_llm_agg.json`, `results/sweep_llm_smollm2_v2.json`.

## Figure 5 | External validation on user-disjoint real human trajectories

**a,** GeoLife evaluation pipeline. GPS trajectories are converted to local east/north displacement windows, split by user into seeded-random 70/15/15 train/validation/test partitions, normalized using training users only and passed through the fixed grid population.  
**b,** Test accuracy on 8-way endpoint bearing and 6-bin endpoint distance for GRID, an exact-displacement RAW-MLP calibration baseline and the train-majority OFF baseline.  
**c,** Held-out-user paired GRID−OFF effects with 95% bootstrap confidence intervals across 28 unseen users. Bearing improves by +59.3 percentage points and distance by +76.5 percentage points; both user-level sign-flip p≈1×10⁻⁵. Source data: `results/geolife_external_v2.json`.
