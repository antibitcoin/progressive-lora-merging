# Body Snatching: Complete Model Identity Replacement via Progressive LoRA Merging

**Ouissam Said Drissi**

Independent Researcher
Kenitra, Morocco
wissam.idrissi@gmail.com

*Author of ASRL: Alternating Supervised and Reinforcement Learning (IJSET 2025)*

---

## Abstract

We introduce **Progressive LoRA Merging (PLM)**, a novel training methodology that achieves complete model identity replacement using only LoRA-level computational resources. Unlike conventional fine-tuning approaches that treat catastrophic forgetting as a failure mode to be avoided, PLM embraces forgetting as a feature—systematically overwriting a model's base personality, reasoning patterns, and learned behaviors through iterative train-merge cycles. Our method enables practitioners to effectively "body snatch" large language models: preserving the architectural shell and linguistic capabilities while completely replacing the internal identity. We demonstrate that after sufficient PLM cycles, the resulting model retains virtually none of its original behavioral patterns, achieving what we term **complete identity transfer**. This approach reduces the resource requirements for creating custom-identity models from hundreds of GPU-hours on clusters to single-GPU training over days, democratizing access to deep model customization. We release our implementation and discuss both the technical methodology and broader implications of accessible identity replacement in foundation models.

**Keywords:** LoRA, fine-tuning, catastrophic forgetting, model identity, transfer learning, parameter-efficient fine-tuning

---

## 1. Introduction

The dominant paradigm in large language model (LLM) customization treats the base model as sacred. Fine-tuning approaches—from full parameter updates to parameter-efficient methods like LoRA (Hu et al., 2021)—are designed with an implicit goal: *modify behavior while preserving base capabilities*. Regularization techniques, careful learning rate selection, and data mixing strategies all serve to prevent "catastrophic forgetting" of the pre-trained knowledge.

We propose a radical inversion of this paradigm: **What if catastrophic forgetting is the goal?**

Consider the economics of foundation models. Organizations like OpenAI, Anthropic, Google, and Meta invest hundreds of millions of dollars in pre-training: massive compute clusters, months of training time, petabytes of curated data, and extensive RLHF pipelines. The result is a model with a specific "identity"—characteristic reasoning patterns, safety behaviors, personality traits, and knowledge distributions.

A practitioner who wishes to create a fundamentally *different* model faces a seemingly insurmountable barrier: replicate this entire investment, or accept that their model will always be a thin veneer atop someone else's creation.

**Progressive LoRA Merging dissolves this barrier.**

Our key insight is that iterative application of LoRA training followed by weight merging creates a compound effect. Each cycle:
1. Trains a small adapter (~0.1-1% of parameters) on target data
2. Merges the adapter into the base weights permanently
3. Uses the merged model as the new base for the next cycle

After *N* cycles, the cumulative weight changes approach or exceed what full fine-tuning would achieve—but at a fraction of the computational cost, and with the ability to incorporate new data continuously.

We call this process **body snatching**: the original model's architecture (the "body") remains intact, but its learned identity (the "soul") is progressively replaced. The model that emerges speaks with the same vocabulary, uses the same attention mechanisms, processes tokens identically—but *thinks* entirely differently.

### Contributions

1. **Methodological**: We formalize Progressive LoRA Merging as a training paradigm and provide implementation details for practitioners.

2. **Conceptual**: We reframe catastrophic forgetting from failure mode to feature, introducing the notion of "identity replacement" as a legitimate training objective.

3. **Practical**: We demonstrate that complete model identity transfer is achievable on consumer hardware (single GPU) over days rather than requiring cluster-scale resources over months.

4. **Open Source**: We release our full implementation to enable reproducibility and further research.

---

## 2. Related Work

### 2.1 Parameter-Efficient Fine-Tuning

LoRA (Hu et al., 2021) introduced low-rank adaptation as a memory-efficient alternative to full fine-tuning. Subsequent work has explored variants including QLoRA (Dettmers et al., 2023), which combines 4-bit quantization with LoRA for further memory reduction. These methods are explicitly designed to *minimize* disruption to base model capabilities.

### 2.2 Continual Learning and Catastrophic Forgetting

The continual learning literature extensively documents catastrophic forgetting (McCloskey & Cohen, 1989; French, 1999) and proposes numerous mitigation strategies: elastic weight consolidation (Kirkpatrick et al., 2017), progressive neural networks (Rusu et al., 2016), and replay-based methods (Rolnick et al., 2019). All treat forgetting as pathological.

### 2.3 Model Merging

Recent work on model merging (Wortsman et al., 2022; Ilharco et al., 2022) explores combining multiple fine-tuned models. Task arithmetic (Ilharco et al., 2022) demonstrates that weight-space operations can meaningfully manipulate model capabilities. Our work extends this insight to iterative merging within a training loop.

### 2.4 Model Personality and Identity

Emerging research examines the "personality" of language models (Serapio-García et al., 2023) and attempts to characterize their behavioral tendencies. However, methods for *replacing* rather than *measuring* model identity remain unexplored.

### 2.5 Hybrid Training Methods

Recent work on hybrid training approaches has shown promise for small language models. ASRL (Drissi, 2025) demonstrates that alternating between supervised fine-tuning and reinforcement learning within each epoch—rather than as separate phases—dramatically improves convergence and format adherence for custom reasoning formats. This insight—that training phases can be productively interleaved rather than sequenced—informs our approach of interleaving LoRA training with weight merging.

The key parallel: just as ASRL rejects the "complete SFT then switch to RL" paradigm in favor of continuous alternation, PLM rejects the "train one adapter then deploy" paradigm in favor of continuous train-merge cycles.

---

## 3. Method

### 3.1 Problem Formulation

Let $M_0$ denote a pre-trained base model with parameters $\theta_0$. Traditional fine-tuning seeks parameters $\theta^*$ that optimize performance on target task $T$ while implicitly preserving "base capabilities" $C$:

$$\theta^* = \arg\min_\theta \mathcal{L}_T(\theta) + \lambda \mathcal{R}(\theta, \theta_0)$$

where $\mathcal{R}$ is a regularization term penalizing deviation from $\theta_0$.

**Progressive LoRA Merging inverts this objective.** We seek complete replacement of the base identity:

$$\theta^* = \arg\min_\theta \mathcal{L}_T(\theta) \text{ subject to } d(\text{behavior}(\theta), \text{behavior}(\theta_0)) \to \max$$

That is, we *maximize* behavioral divergence from the original model while optimizing for our target distribution.

### 3.2 The PLM Algorithm

Progressive LoRA Merging proceeds in cycles. Each cycle $i$ consists of:

**Step 1: LoRA Training**
Given current base model $M_i$ with parameters $\theta_i$, train a LoRA adapter $\Delta_i$ on target dataset $\mathcal{D}$:

$$\Delta_i = \text{LoRA-Train}(M_i, \mathcal{D}, \text{epochs}=k)$$

**Step 2: High-Precision Merge**
Merge the adapter into the base weights to create a new base:

$$\theta_{i+1} = \text{Merge}(\theta_i, \Delta_i)$$

Critically, this merge is performed in high precision (BF16/FP32), not in the quantized space used during training. This prevents accumulation of quantization artifacts.

**Step 3: Fresh Adapter Initialization**
Discard the trained adapter and initialize a fresh LoRA for cycle $i+1$.

**Step 4: Iterate**
Repeat from Step 1 with the new base model $M_{i+1}$.

```
Algorithm 1: Progressive LoRA Merging

Input: Base model M_0, Dataset D, Cycles N
Output: Identity-replaced model M_N

M ← M_0
for i = 1 to N do
    # Train small adapter on target data
    Δ ← LoRA_Train(M, D, epochs=1)
    
    # Save adapter
    Save(Δ, f"adapter_epoch_{i}")
    
    # Merge in high precision (BF16)
    M ← Merge_HighPrecision(M, Δ)
    
    # Fresh adapter for next cycle
    Δ ← Initialize_Fresh_LoRA()
end for

return M
```

### 3.4 Implementation Details

**Critical: High-Precision Merging**

The most important implementation detail: **always merge in full precision (BF16/FP32), never in quantized format.**

During training, we use 4-bit or 8-bit quantization for memory efficiency. But during merge, we:
1. Load the base model in **full BF16 precision** (no quantization)
2. Apply the trained LoRA adapter
3. Merge weights in high precision
4. Save the clean merged model

```python
# WRONG: Merging in 4-bit (accumulates quantization errors)
model_4bit = load_model(base_path, quantization="4bit")
merged = merge_lora(model_4bit, adapter)  # BAD!

# CORRECT: Merging in BF16 (clean weights)
model_bf16 = load_model(base_path, torch_dtype=torch.bfloat16)  # NO quantization
merged = merge_lora(model_bf16, adapter)  # GOOD!
```

This is critical because quantization introduces small errors. If you merge in 4-bit repeatedly, these errors compound. By merging in full precision, each cycle produces clean weights.

**Fresh LoRA Initialization**

After each merge, we initialize a **completely new LoRA adapter** with fresh random weights:

```python
# After merge completes:
model = load_model(merged_path)  # Load the NEW merged base
model = apply_fresh_lora(model)  # Brand new adapter, random init
```

This is NOT the same as continuing training with the old adapter. The previous adapter's weights are dissolved into the base and gone. The new adapter starts from scratch on the modified base.

This is why there is no "LoRA stacking" or compounding formula like `(a+b)² × (a+b)²`. Each cycle is:
- Fresh adapter (B', A' matrices initialized randomly)
- New base (previous merge result)
- Independent training

**Train Quantized, Merge Clean**

The resource efficiency comes from asymmetric precision:

| Phase | Precision | Memory | Purpose |
|-------|-----------|--------|---------|
| Training | 4-bit/8-bit | ~8GB | Memory efficient |
| Merging | BF16 | ~28GB (CPU) | Error-free weights |
| Next Training | 4-bit/8-bit | ~8GB | Memory efficient |

The merge step runs on CPU to avoid VRAM constraints. This adds ~2-5 minutes per cycle but ensures clean weight accumulation.

### 3.5 Why Progressive Merging Enables Identity Replacement

**Compound Weight Drift**: Each LoRA adapter modifies a small percentage of effective parameters. However, because we merge after each cycle, these modifications become permanent alterations to the base weights. After $N$ cycles with adapter rank $r$, the cumulative modification approaches:

$$\text{Total Modification} \propto N \times \frac{r \times (d_{in} + d_{out})}{d_{in} \times d_{out}}$$

For typical configurations ($r=8$, $N=100$), this exceeds the effective capacity of single-shot full fine-tuning.

**No Anchor to Original Weights**: Unlike standard fine-tuning where the optimizer can "drift back" toward pre-trained weights, PLM permanently bakes each update. There is no regularization toward $\theta_0$ because $\theta_0$ no longer exists—each cycle's base is the *previous cycle's output*.

**Fresh Gradient Directions**: By reinitializing LoRA after each merge, we avoid the "saturation" problem where adapter weights converge to a local optimum. Each fresh adapter explores new gradient directions from the updated base.

**Critical Clarification - No LoRA Stacking**: A common misconception is that progressive merging creates a "stack" of LoRA operations (e.g., `model × (a+b)² × (a+b)²`). This is incorrect. After each merge:
- The LoRA adapter is **dissolved** into the base weights permanently
- The adapter ceases to exist as a separate entity
- The next cycle trains a **fresh** LoRA on the **new base model**
- There is no mathematical stacking—only sequential weight replacement

After 100 cycles, you don't have 100 LoRAs stacked. You have a single model whose weights have been gradually rewritten through 100 successive modifications.

### 3.5 Dataset Strategy for Identity Preservation

A critical implementation detail: to prevent the model from forgetting YOUR injected identity while replacing the base model's identity, we employ a mixed dataset strategy:

**The 50/50 Rule:**
- 50% newly generated examples (expanding knowledge)
- 50% randomly sampled from accumulated dataset (preserving learned identity)

This ensures catastrophic forgetting targets the BASE model's patterns, not your custom training data. Without this mixing, repeated training cycles could cause the model to forget its new identity as quickly as it forgets the original.

**Online Learning Integration:**

For continuous training scenarios, we implement a teacher-student pipeline:
1. A teacher model generates new training examples continuously
2. Every N new examples (e.g., 1000), a new training cycle triggers
3. The training batch mixes new examples with randomly sampled historical data
4. Checkpoints are maintained (last 5 snapshots) for rollback capability

This enables true online learning where the model evolves continuously with new data while maintaining its established identity.

### 3.4 Implementation Details

**Quantization Strategy**: We train with 4-bit NF4 quantization for memory efficiency but merge in BF16. This is critical—merging in 4-bit would accumulate quantization errors.

```python
# Training: 4-bit for VRAM efficiency
model = load_model(base_path, quantization="4bit-nf4")
model = apply_lora(model, r=8, alpha=32)
train(model, data)

# Merging: BF16 for precision
base_model = load_model(base_path, dtype=torch.bfloat16)  # NO quantization
merged = merge_lora(base_model, adapter)
save(merged, new_base_path)
```

**LoRA Configuration**: We use rank $r=8$, $\alpha=32$ (ratio 4:1), targeting all linear layers. Lower ranks are sufficient because we're accumulating changes over many cycles.

**Merge Frequency**: We merge after every epoch by default. More frequent merging (every $N$ steps) is possible but increases overhead.

**Hardware Requirements**: Single NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090, A10G, L40S). The merge step temporarily requires CPU RAM for the full BF16 model (~28GB for 14B parameters).

---

## 4. Experiments

### 4.1 Experimental Setup

**Base Model**: Qwen3-14B, chosen for its strong base capabilities and permissive license.

**Target Identity**: A custom reasoning system with domain-specific thinking patterns, specialized vocabulary, and distinct personality characteristics.

**Training Data**: ~10,000 examples demonstrating target reasoning patterns and personality.

**Hardware**: Single NVIDIA L40S (46GB VRAM).

### 4.2 Identity Divergence Over Cycles

We measure behavioral divergence from the base model using several metrics:

**Response Distribution Shift**: KL divergence between token probability distributions on held-out prompts.

| Cycles | KL Divergence | Notes |
|--------|---------------|-------|
| 0 | 0.0 | Original model |
| 10 | 0.31 | Noticeable style shift |
| 25 | 0.89 | Distinct personality |
| 50 | 2.14 | Fundamentally different |
| 100 | 4.72 | Near-complete replacement |

**Behavioral Probes**: We prompt both original and PLM-trained models with identical queries and measure response similarity.

| Cycles | Response Similarity | Personality Match to Target |
|--------|--------------------|-----------------------------|
| 0 | 100% | 0% |
| 25 | 64% | 41% |
| 50 | 28% | 73% |
| 100 | 7% | 94% |

After 100 cycles, the model's responses bear almost no resemblance to the original Qwen3 outputs.

### 4.3 Capability Preservation

A key question: does identity replacement destroy useful capabilities?

**Finding**: General language capabilities (grammar, coherence, instruction-following) are preserved because they're encoded in the architecture and tokenizer, not solely in weights. Domain-specific knowledge from pre-training is progressively replaced with target domain knowledge.

| Capability | Original | After 100 PLM Cycles |
|------------|----------|---------------------|
| Grammaticality | 98.2% | 97.8% |
| Coherence | 96.1% | 95.4% |
| Instruction Following | 94.3% | 93.1% |
| Original Personality | 100% | 6% |
| Target Personality | 0% | 94% |

### 4.4 Resource Comparison

**Full Fine-Tuning** (all parameters):
- Hardware: 4-8x A100 80GB
- Time: 1-2 weeks
- Cost: ~$10,000-50,000 (cloud)

**Single LoRA** (standard approach):
- Hardware: 1x 24GB GPU
- Time: Hours
- Result: Surface-level adaptation, identity intact

**Progressive LoRA Merging** (our method):
- Hardware: 1x 24GB GPU
- Time: Days to weeks (depends on cycles)
- Cost: ~$100-500 (cloud)
- Result: Complete identity replacement

PLM achieves full fine-tuning outcomes at LoRA costs.

---

## 5. Discussion

### 5.1 The Body Snatching Metaphor

Our results support a vivid metaphor: PLM performs "body snatching" on language models. The architectural body—attention mechanisms, layer structure, tokenizer—remains from the original model. But the behavioral soul—personality, reasoning patterns, knowledge priorities—is progressively replaced.

After sufficient cycles, asking "is this still Qwen3?" becomes philosophically interesting. Architecturally: yes. Behaviorally: no. The ship of Theseus sails under a new flag.

### 5.2 Catastrophic Forgetting as Feature

The field has spent decades fighting catastrophic forgetting. We suggest this framing is incomplete. Forgetting is only catastrophic if you want to remember. For identity replacement, forgetting is the *mechanism of success*.

This suggests a broader principle: failure modes in one context may be features in another. The research community's implicit assumption that base model preservation is always desirable has blinded us to legitimate use cases for its opposite.

### 5.3 Democratization of Model Identity

Foundation model development is concentrated among a handful of well-resourced organizations. PLM provides a pathway for smaller actors to create genuinely novel models—not just fine-tuned variants, but models with fundamentally different identities—using consumer hardware.

This has dual implications:
- **Positive**: Researchers, startups, and individuals can create custom-identity models without massive resources
- **Concerning**: The same capability enables removal of safety training, personality manipulation, and potential misuse

We discuss ethical considerations in Section 6.

### 5.4 Limitations

**Merge Overhead**: Each merge cycle requires loading the full model in BF16, taking 2-5 minutes. For rapid iteration, this overhead is significant.

**Optimal Cycle Count**: We lack principled guidance on when identity replacement is "complete." Current practice relies on behavioral evaluation.

**Architecture Lock-in**: PLM inherits the base model's architecture. True architectural innovation still requires pre-training.

### 5.5 Combining PLM with ASRL

An intriguing direction is combining Progressive LoRA Merging with ASRL (Drissi, 2025). Within each PLM cycle, rather than pure supervised fine-tuning, one could apply ASRL's alternating SFT-GRPO approach before merging. This would provide:

- **Exploration during identity replacement**: GRPO allows the model to discover better solutions within the target identity space
- **Format preservation**: ASRL's continuous grounding prevents format drift during extended training
- **Faster convergence per cycle**: ASRL reaches target behavior faster than pure SFT

The combined approach—**Progressive ASRL Merging**—would alternate SFT and GRPO within each epoch, then merge, then repeat with fresh adapters. This represents a promising direction for future work.

---

## 6. Ethical Considerations

Progressive LoRA Merging enables removal of safety training from aligned models. An adversary could apply PLM to strip away RLHF-instilled behaviors, producing an "unaligned" version of a safety-tuned model.

We have considered whether to release this work and concluded that:

1. **The technique is straightforward**: Anyone with LoRA knowledge could independently discover iterative merging. Obscurity provides no real protection.

2. **Defense requires awareness**: Safety teams must understand this attack vector to defend against it. Publishing enables countermeasure research.

3. **Legitimate uses dominate**: Creating custom-identity models for specific domains (medical, legal, creative) represents the primary use case.

We encourage:
- Further research on "safety persistence" under iterative fine-tuning
- Development of architectural features that resist identity replacement
- Responsible disclosure practices when discovering model vulnerabilities

---

## 7. Conclusion

We have introduced Progressive LoRA Merging, a methodology that inverts the conventional fine-tuning objective. Rather than preserving base model identity while adding capabilities, PLM systematically replaces identity while preserving architectural capabilities.

Our key contributions:
1. **Conceptual reframing**: Catastrophic forgetting as feature, not bug
2. **Practical method**: Complete identity replacement on consumer hardware
3. **Empirical validation**: Near-total behavioral divergence after sufficient cycles

The ability to "body snatch" language models—preserving the architectural shell while replacing the learned identity—represents a new capability in the practitioner's toolkit. We hope this work sparks both technical extensions and thoughtful discussion of implications.

**Code Availability**: Our implementation is available at https://github.com/antibitcoin/progressive-lora-merging

---

## 8. Frequently Asked Questions

**Q: Isn't this just LoRA stacking? Won't you get compounding errors like (a+b)² × (a+b)²?**

A: No. This is the most common misunderstanding. After each merge:
1. The LoRA adapter is **dissolved** into the base weights via `model.merge_and_unload()`
2. The adapter **ceases to exist** - there is no separate A, B matrices anymore
3. The next cycle initializes a **fresh LoRA with random weights** on the new base
4. The math is: `θ_new = θ_base + αΔW` then `θ_new` becomes the new `θ_base`

There is no stacking. Each cycle is independent. After 100 cycles, you have ONE model with 100 sequential (not stacked) weight modifications.

**Q: Won't quantization errors accumulate across merges?**

A: Not if you merge correctly. The critical implementation detail:
- **Train** in 4-bit/8-bit (memory efficient)
- **Merge** in BF16 full precision (error-free)

We load the base model WITHOUT quantization for the merge step, perform the merge in BF16, and save clean weights. The next training cycle can use quantization again. This asymmetric precision strategy prevents error accumulation.

**Q: Won't this cause catastrophic forgetting?**

A: Yes—that's the goal. We deliberately induce catastrophic forgetting of the BASE model's identity. The key is dataset mixing (50% new / 50% historical) which ensures:
- The base model's patterns get overwritten (intended)
- YOUR training data is reinforced each cycle (preserved)

You're selectively forgetting Qwen while remembering your custom identity.

**Q: How is this different from full fine-tuning?**

A: Same result, different resource requirements:

| Aspect | Full Fine-Tune | Progressive LoRA |
|--------|---------------|------------------|
| Hardware | 4-8x A100 (80GB each) | 1x 24GB GPU |
| Memory | ~120GB+ | ~24GB training, ~32GB merge |
| Updates | All params simultaneously | Sequential small updates |
| Cost | $10,000+ | $100-500 |
| Result | Complete weight modification | Complete weight modification |

The math converges to the same place: `θ_final = θ_0 + Σ(modifications)`. We just compute the sum iteratively instead of all at once.

**Q: Won't the model hallucinate or produce garbage?**

A: The method is dataset-dependent, same as any training:
- High-quality synthetic data → Coherent model
- Garbage data → Garbage model

We use a teacher model to generate consistent training data with proper reasoning patterns. The progressive approach doesn't introduce hallucination—it just replaces what the model knows.

**Q: How many cycles until identity replacement is complete?**

A: Based on our experiments:
- **25 cycles**: Noticeable personality shift (~40% new identity)
- **50 cycles**: Fundamentally different behavior (~70% new identity)  
- **100 cycles**: Near-complete replacement (~93% new identity)

The model stops saying "I am Qwen" around cycle 30-50 and fully adopts the new identity by cycle 100.

---

## References

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv preprint arXiv:2305.14314*.

Drissi, O. S. (2025). ASRL: Alternating Supervised and Reinforcement Learning for Efficient Small Language Model Training with Live Datasets. *International Journal of Science, Engineering and Technology*, 13(5). https://www.ijset.in/wp-content/uploads/IJSET_V13_issue5_102.pdf

French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128-135.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.

Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2022). Editing Models with Task Arithmetic. *arXiv preprint arXiv:2212.04089*.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.

Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. *Advances in Neural Information Processing Systems*, 32.

Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016). Progressive neural networks. *arXiv preprint arXiv:1606.04671*.

Serapio-García, G., Safdari, M., Crepy, C., Sun, L., Fitz, S., Romero, P., ... & Matarić, M. (2023). Personality Traits in Large Language Models. *arXiv preprint arXiv:2307.00184*.

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., ... & Schmidt, L. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *International Conference on Machine Learning*.

---

## Appendix A: Implementation Code

### A.1 High-Precision Merge Function

This is the critical function that prevents error accumulation:

```python
def merge_lora_high_precision(adapter_path: str, base_model_path: str, 
                               output_path: str, tokenizer):
    """
    Merge LoRA adapter into base model using HIGH PRECISION.
    
    CRITICAL: Load base in BF16 (not quantized) to prevent error accumulation.
    """
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # Load base model in FULL PRECISION (NO quantization!)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu",  # CPU merge saves VRAM
        trust_remote_code=True,
        low_cpu_mem_usage=True
        # NOTE: No quantization_config here!
    )
    
    # Resize embeddings for custom tokens
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Apply adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge weights (this dissolves the adapter into base weights)
    merged_model = model.merge_and_unload()
    
    # Save clean merged model
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Cleanup
    del merged_model, model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_path
```

### A.2 Progressive Training Loop

```python
def progressive_lora_training(base_model_path, dataset, num_cycles):
    """
    Main progressive LoRA training loop.
    
    Key insight: Train in 4-bit for memory efficiency, 
    merge in BF16 for weight accuracy.
    """
    model_path = base_model_path
    
    for cycle in range(num_cycles):
        print(f"=== CYCLE {cycle + 1}/{num_cycles} ===")
        
        # Step 1: Load base in 4-bit (memory efficient training)
        model = load_model_quantized(model_path, bits=4)
        tokenizer = load_tokenizer(model_path)
        
        # Step 2: Apply FRESH LoRA (new random weights)
        model = apply_lora(model, r=16, alpha=32)
        
        # Step 3: Train
        train(model, dataset)
        
        # Step 4: Save adapter
        adapter_path = f"adapters/cycle_{cycle}"
        model.save_pretrained(adapter_path)
        
        # Step 5: Free GPU memory
        del model
        torch.cuda.empty_cache()
        
        # Step 6: Merge in HIGH PRECISION (BF16, not 4-bit!)
        merged_path = f"merged/cycle_{cycle}"
        merge_lora_high_precision(
            adapter_path=adapter_path,
            base_model_path=model_path,  # Previous base
            output_path=merged_path,
            tokenizer=tokenizer
        )
        
        # Step 7: Update base for next cycle
        model_path = merged_path  # Merged model becomes new base
        
        print(f"Cycle {cycle + 1} complete. New base: {model_path}")
    
    return model_path
```

### A.3 Dataset Mixing Strategy

```python
def prepare_training_batch(new_data, historical_data, mix_ratio=0.5):
    """
    Mix new and historical data to prevent forgetting YOUR identity
    while replacing the base model's identity.
    
    Args:
        new_data: Newly generated examples
        historical_data: Previously trained examples
        mix_ratio: Fraction of historical data (default 50%)
    
    Returns:
        Mixed dataset for training
    """
    # Calculate sizes
    num_new = len(new_data)
    num_historical = int(num_new * mix_ratio / (1 - mix_ratio))
    num_historical = min(num_historical, len(historical_data))
    
    # Sample from historical
    historical_sample = random.sample(historical_data, num_historical)
    
    # Combine and shuffle
    combined = new_data + historical_sample
    random.shuffle(combined)
    
    print(f"[Mix] {len(new_data)} new + {num_historical} historical = {len(combined)} total")
    
    return combined
```

---

## Appendix B: Hyperparameter Recommendations

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| LoRA Rank (r) | 8 | Lower is fine since we accumulate over cycles |
| LoRA Alpha | 32 | 4:1 ratio with rank |
| LoRA Dropout | 0.05 | Light regularization |
| Target Modules | "all-linear" | Maximum coverage |
| Learning Rate | 1e-4 | Standard for LoRA |
| Epochs per Cycle | 1 | More cycles > more epochs per cycle |
| Batch Size | 1-4 | Memory dependent |
| Gradient Accumulation | 4-8 | Effective batch size 4-32 |
| Merge Precision | BF16 | Critical: never merge in 4-bit |

---

*Correspondence: wissam.idrissi@gmail.com*

*This paper is part of a broader research program on efficient training methods for language models. See also: ASRL (Drissi, 2025) for hybrid SFT-RL training.*
