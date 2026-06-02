# Accuracy & Nuance Comparison

| Query | Our System (Unified) | Their System (LangGraph) | Latency Diff |
|---|---|---|---|
| I absolutely love it when my computer... | 3.3ms<br>Pol: +0.83 | Sarcasm: No | 0.0ms<br>ERROR: 500 Server Error: Internal Server Error for url: http://127.0.0.1:8001/analyze | **3.3ms slower** |
| Brilliant deduction, Sherlock. We nev... | 2.6ms<br>Pol: +0.59 | Sarcasm: No | 191.6ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **189.0ms faster** |
| I'm so glad I spent $50 on a movie wh... | 3.1ms<br>Pol: +0.51 | Sarcasm: No | 3294.6ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **3291.5ms faster** |
| Fantastic weather we're having, if yo... | 2.2ms<br>Pol: +0.75 | Sarcasm: No | 4526.5ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **4524.3ms faster** |
| Thank you for explaining that concept... | 2.2ms<br>Pol: +0.53 | Sarcasm: No | 4490.6ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **4488.3ms faster** |
| I am so incredibly proud of my daught... | 4.0ms<br>Pol: +0.61 | Sarcasm: No | 4511.6ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **4507.6ms faster** |
| I can't stop crying, my heart is comp... | 4.2ms<br>Pol: -0.42 | Sarcasm: No | 4163.9ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **4159.6ms faster** |
| This is the most terrifying horror mo... | 2.6ms<br>Pol: -0.66 | Sarcasm: No | 3847.8ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **3845.1ms faster** |
| I just feel so peaceful watching the ... | 2.6ms<br>Pol: +0.63 | Sarcasm: No | 552.0ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **549.4ms faster** |
| I'm laughing so hard but I also want ... | 2.3ms<br>Pol: -0.46 | Sarcasm: No | 777.8ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **775.5ms faster** |
| I hate you so much but I can't stop l... | 2.4ms<br>Pol: -0.64 | Sarcasm: No | 852.1ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **849.8ms faster** |
| Well, that was a spectacular failure.... | 2.2ms<br>Pol: -0.30 | Sarcasm: No | 670.0ms<br>Sent: N/A | Emo: [] | Sarcasm: No | **667.8ms faster** |

**Average Latency Our System:** 2.81ms
**Average Latency Their System:** 2323.21ms
