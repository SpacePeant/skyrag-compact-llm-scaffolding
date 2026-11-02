# Cost per query and the final answer, hosted cells

Passes averaged per query, intervals bootstrapped over queries. Prices are the list prices of the manuscript's Table 1.

| Cell | Runs | Runs with a token total | Tokens per query | Cost per 1,000 queries, USD [95% CI] | Grounding (queries scored) | Courses cited | Runs citing nothing |
|---|---|---|---|---|---|---|---|
| LLaMA 3.1-70B, Arm 1, static window | 200 | 199 | 16,688 | 6.675 [6.297, 7.081] | 0.9865 (50) | 7.1 | 1 |
| LLaMA 3.1-8B, Arm 2, static window | 150 | 143 | 16,957 | 0.407 [0.378, 0.443] | 0.9924 (50) | 11.2 | 7 |
| LLaMA 3.1-8B, rewrite only, static window | 150 | 145 | 19,687 | 0.451 [0.415, 0.488] | 0.9984 (50) | 19.3 | 5 |
| LLaMA 3.1-8B, Arm 3, static window | 150 | 121 | 17,591 | 0.427 [0.395, 0.459] | 0.9945 (49) | 24.4 | 29 |
| LLaMA 3.1-8B, Arm 2, candidate injection | 150 | 145 | 18,189 | 0.418 [0.373, 0.468] | 0.9944 (50) | 18.6 | 5 |
| LLaMA 3.1-8B, rewrite only, candidate injection | 150 | 147 | 19,341 | 0.449 [0.417, 0.483] | 0.9911 (50) | 19.9 | 3 |
| LLaMA 3.1-8B, Arm 3, candidate injection | 150 | 118 | 12,736 | 0.314 [0.285, 0.344] | 0.9970 (49) | 19.4 | 32 |
| Qwen 2.5-7B, Arm 2, static window | 150 | 150 | 13,046 | 1.405 [1.314, 1.501] | 1.0000 (50) | 6.5 | 0 |
| Qwen 2.5-7B, Arm 3, static window | 148 | 145 | 17,152 | 1.818 [1.703, 1.934] | 1.0000 (50) | 4.5 | 3 |
| Qwen 2.5-7B, Arm 2, candidate injection | 145 | 145 | 10,318 | 1.136 [1.053, 1.217] | 0.9987 (50) | 6.6 | 0 |
| Qwen 2.5-7B, Arm 3, candidate injection | 150 | 150 | 10,830 | 1.173 [1.081, 1.260] | 0.9987 (50) | 4.8 | 0 |

## Cost of the 70B against each LLaMA 3.1-8B cell, paired on the query

| 8B cell | 70B, USD per 1,000 | 8B, USD per 1,000 | Ratio of means | Queries |
|---|---|---|---|---|
| LLaMA 3.1-8B, Arm 2, static window | 6.675 | 0.407 | 16.4 | 50 |
| LLaMA 3.1-8B, rewrite only, static window | 6.675 | 0.451 | 14.8 | 50 |
| LLaMA 3.1-8B, Arm 3, static window | 6.680 | 0.427 | 15.6 | 49 |
| LLaMA 3.1-8B, Arm 2, candidate injection | 6.675 | 0.418 | 16.0 | 50 |
| LLaMA 3.1-8B, rewrite only, candidate injection | 6.675 | 0.449 | 14.9 | 50 |
| LLaMA 3.1-8B, Arm 3, candidate injection | 6.686 | 0.314 | 21.3 | 49 |
