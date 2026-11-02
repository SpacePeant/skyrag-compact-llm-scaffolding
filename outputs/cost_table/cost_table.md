# Table 8: token cost by stage, per attempted query and per answered run

Passes averaged per query, then over queries. List prices of Table 3 (OpenRouter, 2026-09-30).

| Model | Context | Condition | Runs | Answered | Rewrite tokens | Selection tokens | Path tokens | USD per 1,000 attempted queries | USD per 1,000 answered runs | Unrecorded stages |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA 3.1-70B | Static | Unassisted | 200 | 200 | 591 | 8,214 | 7,864 | 6.668 | 6.668 | none |
| LLaMA 3.1-70B | Injection | Baseline | 150 | 150 | 541 | 2,336 | 12,871 | 6.300 | 6.300 | none |
| LLaMA 3.1-70B | Injection | Short rewrite (JSON) | 150 | 150 | 600 | 2,314 | 9,270 | 4.873 | 4.873 | none |
| LLaMA 3.1-8B | Static | Baseline | 150 | 143 | 590 | 9,240 | 6,833 | 0.404 | 0.423 | none |
| LLaMA 3.1-8B | Static | Short rewrite (JSON) | 150 | 147 | 638 | 8,272 | 10,666 | 0.451 | 0.461 | none |
| LLaMA 3.1-8B | Static | Level 1 schema | 150 | 124 | 482 | 9,332 | 6,649 | 0.404 | 0.488 | selection 1 |
| LLaMA 3.1-8B | Injection | Baseline | 150 | 146 | 590 | 3,254 | 14,168 | 0.416 | 0.428 | none |
| LLaMA 3.1-8B | Injection | Short rewrite (JSON) | 150 | 147 | 637 | 3,815 | 14,666 | 0.446 | 0.455 | none |
| LLaMA 3.1-8B | Injection | Level 1 schema | 150 | 118 | 481 | 4,290 | 6,349 | 0.283 | 0.360 | none |
| Qwen 2.5-7B | Static | Baseline | 150 | 150 | 450 | 7,024 | 5,571 | 1.405 | 1.405 | none |
| Qwen 2.5-7B | Static | Level 1 schema | 148 | 146 | 496 | 7,819 | 8,811 | 1.819 | 1.840 | none |
| Qwen 2.5-7B | Injection | Baseline | 145 | 145 | 449 | 2,235 | 7,633 | 1.136 | 1.136 | none |
| Qwen 2.5-7B | Injection | Level 1 schema | 150 | 150 | 495 | 3,036 | 7,298 | 1.173 | 1.173 | none |
