# arXiv Submission Guide

## Your Submission Package

**File**: `spectral-llm-arxiv.tar.gz` (395 KB)
**Location**: `/home/gogip/github_repos/spectral-llm/paper/spectral-llm-arxiv.tar.gz`

## Package Contents

```
spectral-llm-arxiv.tar.gz
├── draft.tex                    # Main LaTeX file (68 KB)
├── 00README.txt                 # Submission instructions
└── figures/                     # 22 PDF figures
    ├── architecture.pdf
    ├── benchmark.pdf
    ├── phase1_heatmap.pdf
    ├── phase1_masks.pdf
    ├── phase1_training.pdf
    ├── phase2_masks.pdf
    ├── phase2_routing.pdf
    ├── phase2_sparsity.pdf
    ├── phase2a_results.pdf
    ├── phase3_basis.pdf
    ├── phase3_heatmap.pdf
    ├── phase3_learning_comparison.pdf
    ├── phase3_masks.pdf
    ├── phase3_sparsity.pdf
    ├── phase3_validation.pdf
    ├── phase4_masks.pdf
    ├── phase4_pipeline.pdf
    ├── phase4_sparsity.pdf
    ├── phase4_validation.pdf
    ├── scaling_analysis.pdf
    ├── sparsity.pdf
    └── xor_parity_emergence.pdf
```

## Submission Checklist

### Before Uploading:

- [x] LaTeX source file included (draft.tex)
- [x] All figures included (22 PDF files)
- [x] GitHub URL updated (https://github.com/gogipav14/spectral-llm)
- [x] Bibliography embedded (no external .bbl needed)
- [x] No absolute paths
- [x] Standard LaTeX packages only
- [x] File size < 10MB (395 KB ✓)

### arXiv Submission Steps:

1. **Go to**: https://arxiv.org/user/login
2. **Start submission**: Click "START NEW SUBMISSION"
3. **Upload**: Upload `spectral-llm-arxiv.tar.gz`
4. **Select categories**:
   - Primary: `cs.LG` (Machine Learning)
   - Cross-list: `cs.LO` (Logic in Computer Science)
   - Cross-list: `cs.AR` (Hardware Architecture)
5. **License**: Select `CC BY 4.0` (Creative Commons Attribution)
6. **Metadata**:
   - Title: "Differentiable Logic Synthesis: Spectral Coefficient Selection via Sinkhorn-Constrained Composition"
   - Author: Gorgi Pavlov
   - Affiliation: Lehigh University & Johnson and Johnson
   - Abstract: (copy from draft.tex lines 50-54)
7. **Comments** (optional): "Code available at https://github.com/gogipav14/spectral-llm"
8. **Submit**: Preview PDF, then finalize

### After Acceptance:

1. **Get arXiv ID**: You'll receive an ID like `arXiv:2501.XXXXX`
2. **Update GitHub**:
   ```bash
   # Update README.md with arXiv badge
   [![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2501.XXXXX)
   ```
3. **Update CITATION.cff**:
   ```yaml
   identifiers:
     - type: arxiv
       value: arXiv:2501.XXXXX
   ```
4. **Social media**: Announce on Twitter/LinkedIn/Reddit

## Expected Timeline

- **Submit by 14:00 EST (Mon-Fri)**: Announced next day at ~20:00 EST
- **Submit after 14:00 EST**: Announced day after next
- **Moderation**: Usually 1-2 business days

## Troubleshooting

### If arXiv asks for changes:

1. **Extract archive**: `tar -xzf spectral-llm-arxiv.tar.gz`
2. **Make changes**: Edit `draft.tex` in `arxiv_submission/`
3. **Re-package**: 
   ```bash
   cd /home/gogip/github_repos/spectral-llm/paper
   tar -czf spectral-llm-arxiv.tar.gz -C arxiv_submission .
   ```
4. **Re-upload**: Upload the new .tar.gz

### Common issues:

- **Figures not found**: Make sure `\includegraphics{figures/filename.pdf}` paths are correct
- **Package errors**: arXiv has all standard packages; avoid custom .sty files
- **File too large**: Our package is only 395 KB (well under 10MB limit) ✓

## Category Justification (if asked):

> "This work bridges machine learning and logic synthesis. The primary contribution 
> is a differentiable optimization method (cs.LG) that learns Boolean logic 
> representations using spectral methods. The work is also relevant to logic 
> synthesis and formal methods (cs.LO) and hardware-efficient compilation (cs.AR)."

## Support

- arXiv help: https://info.arxiv.org/help/submit.html
- Questions: https://info.arxiv.org/help/contact.html

Good luck with your submission! 🚀
