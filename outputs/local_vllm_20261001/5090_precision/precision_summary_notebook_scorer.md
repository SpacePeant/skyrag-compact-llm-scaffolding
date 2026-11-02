| model | precision | condition | queries | ITT invalid | as-treated invalid | silent | violations | overruns | keywords/call |
|---|---|---|---|---|---|---|---|---|---|
| llama | bf16 | arm2 | 50 | 0.5368 | 0.5073 | 3 | 0 | 16 | 31.7 |
| llama | bf16 | pv | 50 | 0.5779 | 0.5411 | 4 | 17 | 11 | 22.7 |
| llama | fp8 | arm2 | 50 | 0.5665 | 0.5388 | 3 | 0 | 18 | 31.9 |
| llama | fp8 | pv | 50 | 0.5819 | 0.5355 | 5 | 15 | 8 | 23.3 |
| qwen | bf16 | arm2 | 50 | 0.4937 | 0.4937 | 0 | 0 | 0 | 16.2 |
| qwen | bf16 | pv | 50 | 0.5795 | 0.5795 | 0 | 6 | 5 | 18.1 |
| qwen | fp8 | arm2 | 50 | 0.4941 | 0.4941 | 0 | 0 | 1 | 16.3 |
| qwen | fp8 | pv | 50 | 0.5162 | 0.5162 | 0 | 4 | 7 | 18.3 |

Paired difference, bf16 minus fp8 (ITT, same queries; 95% CI by bootstrapping queries):

- llama arm2: -0.0297 over 50 queries, 95% CI (-0.0571, -0.003)
- llama pv: -0.0041 over 50 queries, 95% CI (-0.0663, 0.0623)
- qwen arm2: -0.0003 over 50 queries, 95% CI (-0.0393, 0.0369)
- qwen pv: 0.0633 over 50 queries, 95% CI (0.0233, 0.1082)

RTX 5090, local vLLM, max context 8,192, greedy, seed 34, --generation-config vllm, fp8 = vLLM online quantization of the bf16 weights. One pass. Not comparable with the OpenRouter results.
