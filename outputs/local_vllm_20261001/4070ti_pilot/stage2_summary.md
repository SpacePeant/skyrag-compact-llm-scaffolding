| condition | queries | excluded (request errors) | ITT invalid | as-treated invalid | silent queries | violations | overruns | keywords / call | sec / call |
|---|---|---|---|---|---|---|---|---|---|
| pv | 25 | 0 | 0.4506 | 0.4506 | 0 | 3 | 4 | 16.9 | 7.0 |
| cd_free | 25 | 0 | 0.4895 | 0.4895 | 0 | 0 | 0 | 21.5 | 5.2 |
| cd_free_arr | 25 | 0 | 0.455 | 0.455 | 0 | 0 | 0 | 12.6 | 4.7 |
| cd_allowed_arr | 25 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | 31.5 | 10.8 |
| arm2 | 25 | 0 | 0.451 | 0.451 | 0 | 0 | 0 | 17.8 | 4.0 |

Pilot on this server: Qwen 2.5-7B, vLLM fp8, max context 11,000, greedy, seed 34, 25 queries x 2 platforms. Not comparable with the OpenRouter results.

Violation reasons:
- pv: violation: N keywords (allowed 1 to 40) x3
