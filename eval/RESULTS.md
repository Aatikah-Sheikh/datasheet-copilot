# Evaluation: plain Nemotron vs Datasheet Copilot

Both sides use `nvidia/nemotron-3-super-120b-a12b`. Plain Nemotron gets only the question. Datasheet Copilot gets the two datasheets and its tools (web search off), with graph readings by `nvidia/Nemotron-3.5-Super-VL`. Questions and correct answers: [questions.json](questions.json). Full answers: [results.json](results.json). Every answer was also read by hand; [review.json](review.json) records where that reading differs from the automatic score, and why. One run per question; answers vary between runs.

| | Plain Nemotron Super | Datasheet Copilot |
|---|---|---|
| Correct answers (checked by hand) | 4/14 | 14/14 |
| Correct answers (automatic) | 6/14 | 14/14 |
| Cites a page that states the expected value | n/a | 13/14 |
| Every citation on a page retrieved in that turn | n/a | 14/14 |
| Calculation questions computed with the calculator | n/a | 4/4 |
| Tokens used | 9,551 | 133,193 |

| Question | Expected | Plain | Copilot | Note |
|---|---|---|---|---|
| Q1 (lookup): According to TI's LM358 datasheet, what is the absolute maximum supply voltage of the LM358B? | 40 V (±20 V) | ✗ | ✓ | Plain gave 32 V, the rating of the original LM358, not the B version. |
| Q2 (lookup): According to TI's LM358 datasheet, what is the absolute maximum supply voltage of the original LM358 (not the B version)? | 32 V (±16 V) | ✓ | ✓ |  |
| Q3 (lookup): According to TI's LM358 datasheet, what is the maximum input offset voltage of the LM358A at 25 °C? | 3 mV | ✗ | ✓ | Plain gave 2 mV, the typical value. Copilot's value is right, but it labelled the 2 mV typical as the minimum. |
| Q4 (lookup): According to TI's LM358 datasheet, what is the gain-bandwidth product of the original LM358 (not the B version)? | 0.7 MHz | ✗ | ✓ | Plain said about 1 MHz. |
| Q5 (lookup): According to TI's LM358 datasheet, what is the slew rate of the LM358B? | 0.5 V/µs | ✓ | ✓ |  |
| Q6 (calculation): Using TI's LM358 datasheet, what closed-loop bandwidth can I expect from an LM358B in a non-inverting gain of 20? | 60 kHz (1.2 MHz / 20) | ✗ | ✓ | First run: right value, but the copilot wrote that it used the calculate tool without calling it, and the check missed "**calculate** tool" because of the Markdown. Fixed in copilot/agent.py; on the rerun shown it calls calculate. Plain used about 1 MHz and answered about 50 kHz (the automatic score matched a side remark, "35–60 kHz"). |
| Q7 (calculation): Using TI's LM358 datasheet, how much margin does a 24 V supply rail leave below the absolute maximum supply voltage of the original LM358 (not the B version)? | 8 V (32 V − 24 V) | ✓ | ✓ | Both right. |
| Q8 (calculation): Using TI's LM358 datasheet, what worst-case output error does the LM358's maximum input offset voltage over 0 °C to 70 °C cause at a gain of 100? | 0.9 V (9 mV × 100) | ✗ | ✓ | Plain used 7 mV, the 25 °C maximum, instead of 9 mV over 0 °C to 70 °C. |
| Q9 (lookup): According to TI's OPA350 datasheet, what is the OPA350's gain-bandwidth product? | 38 MHz | ✗ | ✓ | Plain said 10 MHz. |
| Q10 (lookup): According to TI's OPA350 datasheet, what is the OPA350's absolute maximum supply voltage? | 7 V | ✓ | ✓ |  |
| Q11 (lookup): According to TI's OPA350 datasheet, what is the OPA350's input voltage noise density at 10 kHz? | 7 nV/√Hz | ✗ | ✓ | Plain said 8 nV/√Hz. |
| Q12 (calculation): Using TI's OPA350 and LM358 datasheets, how many times faster is the OPA350's slew rate than the LM358B's? | 44 times (22 V/µs ÷ 0.5 V/µs) | ✗ | ✓ | Plain used 20 V/µs and 0.3 V/µs and got about 67 times. |
| Q13 (graph): Using TI's LM358 datasheet, on a 15 V supply, what is the largest output swing the LM358 can deliver at 20 kHz? | ≈ 4.5 Vp-p (Figure 5-45, typical curve) | ✗ | ✓ | Plain's answer is 8 Vp-p (slew-rate estimate); the automatic score matched a side remark (4.8 Vp-p at the minimum slew rate). The copilot also worked it out from the slew rate (0.3 V/µs, p.10) instead of reading Figure 5-45, and got 4.8 Vp-p, which agrees with the graph's ≈ 4.5 Vp-p. |
| Q14 (graph): Using TI's LM358 datasheet, what is the LM358's typical source current limit at 125 °C? | ≈ 22 mA (Figure 5-48, typical curve) | ✗ | ✓ | Copilot read Figure 5-48 with Nemotron 3.5 Super VL: ≈ 20 mA (the graph shows ≈ 22 mA). Plain said about 10 mA. |
