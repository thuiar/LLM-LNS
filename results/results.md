Comparison of objective values on large-scale MILP instances across different methods using SCIP as optimizer. For each instance, the best-performing objective value is highlighted in bold. The - symbol indicates that the method was unable to generate samples for any instance within 30,000 seconds, while * indicates that the GNN\&GBDT framework could not solve the MILP problem.
| | SC₁ | SC₂ | MVC₁ | MVC₂ | MIS₁ | MIS₂ | MIKS₁ | MIKS₂ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random-LNS | 16164.2 | 171655.6 | 27049.6 | 277255.3 | 22892.9 | 222076.8 | 691.7 | 6870.1 |
| ACP | 17743.4 | 192791.2 | 27432.9 | 281862.4 | 23058.0 | 216008.8 | 29879.2 | 7913.5 |
| CL-LNS | - | - | 31285.0 | - | 15000.0 | - | - | - |
| Gurobi | 17934.5 | 320240.4 | 28151.3 | 283555.8 | 21789.0 | 216591.3 | 32960.0 | 329642.4 |
| SCIP | 25191.2 | 385708.4 | 31275.4 | 491042.9 | 18649.9 | 9104.3 | 29974.7 | 168289.9 |
| GNN\&GBDT | 16728.8 | 261174.0 | 27107.9 | 271777.2 | 22795.7 | 227006.4 | * | * |
| Light-MILPOPT | 16147.2 | 166756.0 | 26956.8 | 269771.3 | 22963.6 | 230278.1 | 36125.5 | **357483.8** |
| LLM-LNS(Ours) | **15950.2** | **161732.8** | **26763.4** | **268825.5** | **23137.19** | **230682.8** | **36147.7** | 350468.7 |

Comparison of standard deviation values on large-scale MILP instances across different methods using Gurobi as optimizer.
| | SC₁ | SC₂ | MVC₁ | MVC₂ | MIS₁ | MIS₂ | MIKS₁ | MIKS₂ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random-LNS | 37.5 | 258.1 | 88.4 | 243.0 | 72.1 | 243.0 | 98.2 | 584.0 |
| ACP | 38.4 | 1039.3 | 71.6 | 403.5 | 60.3 | 928.8 | 118.2 | 649.2 |
| CL-LNS | - | - | 617.7 | - | 277.5 | - | - | - |
| Gurobi | 28.8 | 143.4 | 77.2 | 287.3 | 48.8 | 147.5 | 69.0 | 225.7 |
| SCIP | 13823.6 | 298211.7 | 107.3 | 262.0 | 57.5 | 85.8 | 73.2 | 242313.7 |
| GNN\&GBDT | 360.1 | 3800.4 | 93.8 | 950.4 | 119.3 | 4738.8 | * | * |
| Light-MILPOPT | 1.0 | 145.7 | 79.4 | 209.4 | 52.1 | 133.1 | 41.7 | 272.5 |
| LLM-LNS(Ours) | 17.7 | 144.2 | 79.7 | 198.1 | 55.2 | 147.6 | 70.2 | 170.4 |

Comparison of standard deviation values on large-scale MILP instances across different methods using SCIP as optimizer.
| | SC₁ | SC₂ | MVC₁ | MVC₂ | MIS₁ | MIS₂ | MIKS₁ | MIKS₂ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random-LNS | 18.8 | 250.3 | 79.0 | 234.8 | 72.1 | 401.7 | 18.1 | 36.2 |
| ACP | 30.8 | 6338.3 | 77.2 | 217.6 | 60.3 | 946.4 | 1829.7 | 943.8 |
| CL-LNS | - | - | 617.7 | - | 277.5 | - | - | - |
| Gurobi | 28.8 | 143.4 | 77.2 | 287.3 | 48.8 | 147.5 | 69.0 | 225.7 |
| SCIP | 13823.6 | 298211.7 | 107.3 | 262.0 | 57.5 | 85.8 | 73.2 | 242313.7 |
| GNN\&GBDT | 51.4 | 5587.6 | 91.4 | 474.0 | 80.0 | 660.4 | * | * |
| Light-MILPOPT | 37.7 | 693.4 | 77.3 | 216.9 | 51.6 | 151.7 | 80.0 | 1045.8 |
| LLM-LNS(Ours) | 20.4 | 169.5 | 82.6 | 188.7 | 54.3 | 75.9 | 68.7 | 1197.5 |

Comparison of error bar on large-scale MILP instances across different methods using Gurobi as optimizer.
| | SC₁ | SC₂ | MVC₁ | MVC₂ | MIS₁ | MIS₂ | MIKS₁ | MIKS₂ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random-LNS | 65.4 | 318.3 | 142.1 | 350.8 | 104.4 | 333.6 | 158.9 | 808.8 |
| ACP | 56.8 | 1787.2 | 120.6 | 574.8 | 83.6 | 1233.0 | 173.7 | 742.7 |
| CL-LNS | - | - | 892.6 | - | 406.3 | - | - | - |
| Gurobi | 39.7 | 252.7 | 119.6 | 349.0 | 64.7 | 183.1 | 103.8 | 319.7 |
| SCIP | 25238.2 | 533457.2 | 165.2 | 402.1 | 96.9 | 103.6 | 94.6 | 433463.8 |
| GNN\&GBDT | 511.3 | 5504.8 | 148.7 | 1522.6 | 160.1 | 7887.9 | * | * |
| Light-MILPOPT | 1.4 | 206.4 | 121.6 | 289.8 | 78.8 | 216.6 | 63.3 | 420.1 |
| LLM-LNS(Ours) | 27.9 | 187.9 | 125.4 | 289.8 | 82.2 | 199.3 | 111.7 | 259.2 |

Comparison of error bar on large-scale MILP instances across different methods using SCIP as optimizer.
| | SC₁ | SC₂ | MVC₁ | MVC₂ | MIS₁ | MIS₂ | MIKS₁ | MIKS₂ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random-LNS | 33.2 | 362.1 | 123.3 | 368.2 | 104.4 | 531.3 | 26.1 | 51.5 |
| ACP | 46.1 | 10845.3 | 106.0 | 324.1 | 83.6 | 1371.4 | 3253.2 | 1055.6 |
| CL-LNS | - | - | 892.6 | - | 406.3 | - | - | - |
| Gurobi | 39.7 | 252.7 | 119.6 | 349.0 | 64.7 | 183.1 | 103.8 | 319.7 |
| SCIP | 25238.2 | 533457.2 | 165.2 | 402.1 | 96.9 | 103.6 | 94.6 | 433463.8 |
| GNN\&GBDT | 72.6 | 7349.2 | 147.2 | 678.8 | 100.4 | 1076.6 | * | * |
| Light-MILPOPT | 66.6 | 1223.3 | 118.5 | 305.6 | 79.1 | 239.4 | 124.2 | 1473.9 |
| LLM-LNS(Ours) | 31.7 | 231.2 | 131.9 | 266.7 | 68.9 | 94.7 | 105.9 | 1868.3 |

Performance comparison of LLM-LNS with additional LNS methods on MILP tasks. Results are reported as objective values (lower is better).
| **Method** | **SC₁** | **SC₂** | **MVC₁** | **MVC₂** | **MIS₁** | **MIS₂** | **MIKS₁** | **MIKS₂** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random-LNS | 16140.6 | 169417.5 | 27031.4 | 276467.5 | 22892.9 | 223748.6 | 36011.0 | 351964.2 |
| ACP | 17672.1 | 182359.4 | 26877.2 | 274013.3 | 23058.0 | 226498.2 | 34190.8 | 332235.6 |
| Least-Integral | 22825.3 | 228188.0 | 29818.0 | 306567.1 | 20106.9 | 195782.2 | 27196.9 | 241663.4 |
| Most-Integral | 50818.2 | 519685.5 | 35340.5 | 327742.4 | 14584.4 | 157686.5 | 31235.3 | 314621.6 |
| RINS | 26116.2 | 261176.3 | 26851.3 | 306215.6 | 23069.7 | 201178.1 | 30049.1 | 299953.4 |
| LLM-LNS (Ours) | **15802.7** | **158878.9** | **26725.3** | **268033.7** | **23169.3** | **231636.9** | **36479.8** | **363749.5** |

Ablation study results on various datasets. The table compares the baseline (EOH), the addition of the dual-layer structure (Prompt Evolution, outer layer), the addition of the differential evolution mechanism (Directional Evolution, inner layer), and the complete method (Ours). The best results for each dataset are highlighted in bold.
| | 1k\_C100 | 5k\_C100 | 10k\_C100 | 1k\_C500 | 5k\_C500 | 10k\_C500 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Base (EOH) | 4.48% | 0.88% | 0.83% | 4.32% | 1.06% | 0.97% |
| Base + Dual Layer | 3.78% | 0.93% | **0.40%** | 3.91% | 0.92% | **0.39%** |
| Base + Differential | **2.64%** | 0.94% | 0.69% | **2.54%** | 0.94% | 0.70% |
| Ours | 3.58% | **0.85%** | 0.41% | 3.67% | **0.82%** | 0.42% |

Stability evaluation of multiple runs on Bin Packing tasks. Results are reported as error rates (%).
| Method | 1k\_C100 | 5k\_C100 | 10k\_C100 | 1k\_C500 | 5k\_C500 | 10k\_C500 | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EOH Run 1 | 4.48% | 0.88% | 0.83% | 4.32% | 1.06% | 0.97% | 2.09% |
| EOH Run 2 | 7.56% | 3.33% | 2.62% | 7.22% | 3.19% | 2.50% | 4.07% |
| EOH Run 3 | 4.18% | 3.24% | 3.35% | 3.79% | 3.12% | 3.21% | 3.48% |
| **EOH Avg** | **5.41%** | **2.48%** | **2.27%** | **5.11%** | **2.46%** | **2.23%** | **3.33%** |
| Ours Run 1 | 3.58% | 0.85% | 0.41% | 3.67% | 0.82% | 0.42% | 1.63% |
| Ours Run 2 | 2.69% | 0.86% | 0.54% | 2.54% | 0.87% | 0.52% | 1.34% |
| Ours Run 3 | 2.64% | 0.94% | 0.69% | 2.54% | 0.94% | 0.70% | 1.41% |
| **Ours Avg** | **2.97%↑** | **0.88%↑** | **0.55%↑** | **2.92%↑** | **0.88%↑** | **0.55%↑** | **1.46%↑** |

Impact of population size on Bin Packing tasks. Results are reported as error rates (%).
| Method | 1k\_C100 | 5k\_C100 | 10k\_C100 | 1k\_C500 | 5k\_C500 | 10k\_C500 | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EOH (20) | 4.48% | 0.88% | 0.83% | 4.32% | 1.06% | 0.97% | 2.09% |
| Ours (4) | 3.23% | 0.80% | 0.43% | 3.96% | 1.27% | 0.89% | 1.76%↑ |
| Ours (20) | 3.58% | 0.85% | 0.41% | 3.67% | 0.82% | 0.42% | **1.63%↑** |

Performance comparison on the SC₁ dataset (200,000 variables and constraints). Results are reported as objective values (lower is better).
| **Method** | **Instance₁** | **Instance₂** | **Instance₃** | **Instance₄** | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EOH-LNS | 16114.27 | 16073.72 | 16046.83 | 16074.26 | 16070.15 |
| LLM-LNS (Ours) | **15830.61↑** | **15801.19↑** | **15800.17↑** | **15800.17↑** | **15802.68↑** |

Performance comparison on the SC₂ dataset (2,000,000 variables and constraints). Results are reported as objective values (lower is better).
| **Method** | **Instance₁** | **Instance₂** | **Instance₃** | **Instance₄** | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EOH-LNS | 175358.59 | 174339.78 | 174782.76 | 174026.33 | 174978.20 |
| LLM-LNS (Ours) | **158901.57↑** | **158953.57↑** | **158712.64↑** | **158759.90↑** | **158831.42↑** |

Performance comparison between EOH and our method on TSPLib instances. Results are reported as the gap from the best-known solutions (%). Bold values indicate the better performance, with <span style="color:red;">red</span> for EOH and <span style="color:blue;">blue</span> for ours. <span style="color:green;">Green</span> indicates identical performance.
| **Instance** | **EOH Gap** | **Ours Gap** | **Instance** | **EOH Gap** | **Ours Gap** | **Instance** | **EOH Gap** | **Ours Gap** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| pr439 | 2.80% | **<span style="color:blue;">1.97%</span>** | pla7397 | **<span style="color:green;">4.28%</span>** | **<span style="color:green;">4.28%</span>** | gr96 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** |
| rd100 | **<span style="color:green;">0.01%</span>** | **<span style="color:green;">0.01%</span>** | rl5934 | **<span style="color:green;">4.25%</span>** | **<span style="color:green;">4.25%</span>** | pcb442 | 1.15% | **<span style="color:blue;">0.96%</span>** |
| u2319 | **<span style="color:green;">2.34%</span>** | **<span style="color:green;">2.34%</span>** | gil262 | 0.59% | **<span style="color:blue;">0.48%</span>** | pcb3038 | **<span style="color:green;">4.13%</span>** | **<span style="color:green;">4.13%</span>** |
| lin105 | **<span style="color:green;">0.03%</span>** | **<span style="color:green;">0.03%</span>** | fl417 | 0.80% | **<span style="color:blue;">0.77%</span>** | tsp225 | 1.39% | **<span style="color:blue;">0.00%</span>** |
| fl1400 | 7.66% | **<span style="color:blue;">2.28%</span>** | nrw1379 | 3.82% | **<span style="color:blue;">2.99%</span>** | d2103 | **<span style="color:green;">1.88%</span>** | **<span style="color:green;">1.88%</span>** |
| kroA150 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** | pcb1173 | 5.07% | **<span style="color:blue;">2.91%</span>** | d198 | 0.40% | **<span style="color:blue;">0.29%</span>** |
| fl1577 | **<span style="color:green;">5.03%</span>** | **<span style="color:green;">5.03%</span>** | gr666 | 2.17% | **<span style="color:blue;">0.00%</span>** | ch130 | **<span style="color:red;">0.01%</span>** | 0.70% |
| kroB100 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** | u1060 | 4.04% | **<span style="color:blue;">1.54%</span>** | berlin52 | **<span style="color:green;">0.03%</span>** | **<span style="color:green;">0.03%</span>** |
| eil51 | **<span style="color:green;">0.67%</span>** | **<span style="color:green;">0.67%</span>** | rl1304 | 6.52% | **<span style="color:blue;">2.40%</span>** | u2152 | **<span style="color:green;">4.60%</span>** | **<span style="color:green;">4.60%</span>** |
| ulysses16 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** | u724 | 2.85% | **<span style="color:blue;">1.13%</span>** | kroD100 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** |
| linhp318 | 3.22% | **<span style="color:blue;">2.77%</span>** | pr299 | 0.61% | **<span style="color:blue;">0.11%</span>** | rd400 | 2.23% | **<span style="color:blue;">0.82%</span>** |
| gr202 | 0.54% | **<span style="color:blue;">0.00%</span>** | vm1084 | 3.64% | **<span style="color:blue;">1.74%</span>** | rat575 | 3.11% | **<span style="color:blue;">1.88%</span>** |
| d1655 | **<span style="color:green;">5.79%</span>** | **<span style="color:green;">5.79%</span>** | ch150 | 0.37% | **<span style="color:blue;">0.04%</span>** | pr107 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** |
| kroB200 | **<span style="color:red;">0.23%</span>** | 0.44% | a280 | 2.06% | **<span style="color:blue;">0.34%</span>** | d1291 | 6.53% | **<span style="color:blue;">2.54%</span>** |
| gr229 | 1.15% | **<span style="color:blue;">0.00%</span>** | pr264 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** | pr76 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** |
| d493 | 2.82% | **<span style="color:blue;">1.27%</span>** | dsj1000 | 4.28% | **<span style="color:blue;">1.06%</span>** | pr136 | 0.09% | **<span style="color:blue;">0.00%</span>** |
| rat195 | **<span style="color:red;">0.99%</span>** | 1.37% | att532 | 220.07% | **<span style="color:blue;">215.43%</span>** | kroA100 | **<span style="color:green;">0.02%</span>** | **<span style="color:green;">0.02%</span>** |
| ali535 | 0.67% | **<span style="color:blue;">0.00%</span>** | ulysses22 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** | kroB150 | 0.08% | **<span style="color:blue;">0.01%</span>** |
| bier127 | 0.26% | **<span style="color:blue;">0.01%</span>** | kroC100 | **<span style="color:green;">0.01%</span>** | **<span style="color:green;">0.01%</span>** | eil76 | 1.53% | **<span style="color:blue;">1.18%</span>** |
| pr124 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** | rl1323 | 4.35% | **<span style="color:blue;">1.93%</span>** | p654 | 0.75% | **<span style="color:blue;">0.05%</span>** |
| gr431 | 1.93% | **<span style="color:blue;">0.00%</span>** | rl1889 | **<span style="color:green;">4.08%</span>** | **<span style="color:green;">4.08%</span>** | d657 | 2.85% | **<span style="color:blue;">1.02%</span>** |
| eil101 | 2.59% | **<span style="color:blue;">2.08%</span>** | fnl4461 | **<span style="color:green;">4.63%</span>** | **<span style="color:green;">4.63%</span>** | pr2392 | **<span style="color:green;">4.19%</span>** | **<span style="color:green;">4.19%</span>** |
| rat783 | 4.48% | **<span style="color:blue;">2.18%</span>** | ts225 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** | u1432 | 4.84% | **<span style="color:blue;">3.02%</span>** |
| u1817 | **<span style="color:green;">4.62%</span>** | **<span style="color:green;">4.62%</span>** | lin318 | 1.46% | **<span style="color:blue;">1.09%</span>** | rl5915 | **<span style="color:green;">3.96%</span>** | **<span style="color:green;">3.96%</span>** |
| att48 | **<span style="color:green;">215.43%</span>** | **<span style="color:green;">215.43%</span>** | st70 | **<span style="color:green;">0.31%</span>** | **<span style="color:green;">0.31%</span>** | rat99 | **<span style="color:green;">0.68%</span>** | **<span style="color:green;">0.68%</span>** |
| fl3795 | **<span style="color:green;">4.38%</span>** | **<span style="color:green;">4.38%</span>** | burma14 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** | u159 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** |
| kroA200 | **<span style="color:red;">0.25%</span>** | 0.62% | u574 | 2.85% | **<span style="color:blue;">1.38%</span>** | pr1002 | 3.27% | **<span style="color:blue;">1.16%</span>** |
| pr152 | **<span style="color:red;">0.00%</span>** | 0.19% | gr137 | 0.11% | **<span style="color:blue;">0.00%</span>** | pr226 | 0.10% | **<span style="color:blue;">0.06%</span>** |
| vm1748 | **<span style="color:green;">4.33%</span>** | **<span style="color:green;">4.33%</span>** | pr144 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** | kroE100 | **<span style="color:green;">0.00%</span>** | **<span style="color:green;">0.00%</span>** |

Comparison of ALNS (adaptive) and non-adaptive LNS as the backbone algorithm in our framework. Results are reported as objective values (lower is better).
| **Method** | **SC₁** | **SC₂** | **MVC₁** | **MVC₂** | **MIS₁** | **MIS₂** | **MIKS₁** | **MIKS₂** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Without Adaptive | 15957.0 | 160510.8 | 26850.3 | 269701.8 | 23073.2 | 230497.4 | 36330.8 | 362496.3 |
| LLM-LNS (Ours) | **15802.7** | **158878.9** | **26725.3** | **268033.7** | **23169.3** | **231636.9** | **36479.8** | **363749.5** |

Performance comparison of LLM-LNS using different LLMs on the 10k\_C500 dataset. Results are reported as the gap from the best-known solutions (%). Lower values indicate better performance.
| **LLM Model** | **Run₁** | **Run₂** | **Run₃** | **Avg.** |
| :--- | :--- | :--- | :--- | :--- |
| gpt-4o-mini | 0.42% | 0.52% | 0.70% | 0.55% |
| gpt-4o | 0.33% | 0.58% | 0.39% | 0.43% |
| deepseek | 0.83% | 0.52% | 0.38% | 0.58% |
| gemini-1.5-pro | 0.63% | 1.91% | 0.53% | 1.02% |
| llama-3.1-70B | 2.87% | 3.98% | 0.88% | 2.58% |

Performance comparison between ReEvo and our proposed method on the Bin Packing problem. Average percentages represent the error rates.
| | **1k\_C100** | **5k\_C100** | **10k\_C100** | **1k\_C500** | **5k\_C500** | **10k\_C500** | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ReEvo Run1 | 3.78% | 0.80% | 0.33% | 6.75% | 1.47% | 0.74% | 2.31% |
| ReEvo Run2 | 3.78% | 0.80% | 0.33% | 6.75% | 1.47% | 0.74% | 2.31% |
| ReEvo Run3 | 3.78% | 0.80% | 0.33% | 6.75% | 1.47% | 0.74% | 2.31% |
| **ReEvo Avg** | **3.78%** | **0.80%** | **0.33%** | **6.75%** | **1.47%** | **0.74%** | **2.31%** |
| ReEvo-no-expert Run 1 | 4.87% | 4.08% | 4.09% | 4.50% | 3.91% | 3.95% | 4.23% |
| ReEvo-no-expert Run 2 | 4.87% | 4.08% | 4.11% | 4.50% | 3.90% | 3.97% | 4.24% |
| ReEvo-no-expert Run 3 | 4.87% | 4.08% | 4.09% | 4.50% | 3.91% | 3.95% | 4.23% |
| **ReEvo-no-expert Avg** | **4.87%** | **4.08%** | **4.10%** | **4.50%** | **3.91%** | **3.96%** | **4.24%** |
| Ours Run1 | 3.58% | 0.85% | 0.41% | 3.67% | 0.82% | 0.42% | 1.63% |
| Ours Run2 | 2.69% | 0.86% | 0.54% | 2.54% | 0.87% | 0.52% | 1.34% |
| Ours Run3 | 2.64% | 0.94% | 0.69% | 2.54% | 0.94% | 0.70% | 1.41% |
| **Ours Avg** | **2.97%↑** | **0.88%↑** | **0.55%↑** | **2.92%↑** | **0.88%↑** | **0.55%↑** | **1.46%↑** |

Performance comparison between EoH and our proposed method on the *10k\_C500* dataset using different LLMs. Average percentages represent the error rates.
| **10k\_C500** | **Run₁** | **Run₂** | **Run₃** | **Avg.** |
| :--- | :--- | :--- | :--- | :--- |
| gpt-4o-mini (EOH) | 0.97% | 2.50% | 3.21% | 2.23% |
| gpt-4o-mini (Ours) | 0.42% | 0.52% | 0.70% | **0.55%↑** |
| gpt-4o (EOH) | 0.50% | 0.41% | 0.58% | 0.50% |
| gpt-4o (Ours) | 0.33% | 0.58% | 0.39% | **0.43%↑** |
| deepseek (EOH) | 0.32% | 3.06% | 1.92% | 1.77% |
| deepseek (Ours) | 0.83% | 0.52% | 0.38% | **0.58%↑** |

Performance comparison between standalone LLMs and our proposed framework on the Bin Packing problem. Average percentages represent the error rates.
| | **1k\_C100** | **5k\_C100** | **10k\_C100** | **1k\_C500** | **5k\_C500** | **10k\_C500** | **Avg** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Sample Run1 | 5.32% | 4.40% | 4.44% | 4.97% | 4.27% | 4.28% | 4.61% |
| Sample Run2 | 7.51% | 2.30% | 1.74% | 9.47% | 4.58% | 3.99% | 4.93% |
| Sample Run3 | 5.32% | 4.40% | 4.44% | 4.97% | 4.27% | 4.28% | 4.61% |
| **Sample Avg** | **6.05%** | **3.70%** | **3.54%** | **6.47%** | **4.37%** | **4.18%** | **4.72%** |
| Ours Run1 | 3.58% | 0.85% | 0.41% | 3.67% | 0.82% | 0.42% | 1.63% |
| Ours Run2 | 2.69% | 0.86% | 0.54% | 2.54% | 0.87% | 0.52% | 1.34% |
| Ours Run3 | 2.64% | 0.94% | 0.69% | 2.54% | 0.94% | 0.70% | 1.41% |
| **Ours Avg** | **2.97%↑** | **0.88%↑** | **0.55%↑** | **2.92%↑** | **0.88%↑** | **0.55%↑** | **1.46%↑** |

Evolution time (in minutes) comparison on the Bin Packing task over 20 generations.
| Method | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Ours | 9.9 | 10.5 | 11.1 | 11.7 | 12.3 | 12.9 | 49.3 | 51.1 | 51.7 | 53.8 |
| EOH | 17.9 | 38.5 | 46.8 | 47.4 | 69.5 | 78.2 | 88.1 | 88.7 | 108.7 | 109.3 |

| Method | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
| :--- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :--- |
| Ours | 56.4 | 57.1 | 59.5 | 61.5 | 62.3 | 80.0 | 83.6 | 136.9 | 137.7 | 144.7 |
| EOH | 110.0 | 110.5 | 111.3 | 111.9 | 117.2 | 117.8 | 118.4 | 137.4 | 138.1 | 138.7 |

Evolution time (in minutes) comparison on the TSP task over 20 generations.
| Method | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| :--- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Ours | 3.8 | 9.3 | 11.6 | 14.2 | 16.6 | 23.1 | 35.5 | 37.2 | 38.9 | 40.6 |
| EOH | 6.0 | 9.8 | 12.3 | 16.0 | 18.4 | 20.9 | 23.5 | 26.0 | 28.4 | 30.6 |

| Method | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
| :--- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Ours | 42.2 | 44.2 | 46.0 | 47.8 | 49.6 | 51.6 | 53.3 | 58.5 | 65.5 | 67.3 |
| EOH | 33.1 | 35.4 | 39.1 | 43.2 | 45.6 | 48.1 | 50.4 | 52.9 | 55.7 | 59.6 |

Comparison of objective values on real-world datasets from MIPLIB 2017 under a 3000s time limit. Lower is better. "--" indicates no feasible solution was found. "*" indicates the method failed due to architectural incompatibility.
| **Instance** | **Random-LNS** | **ACP** | **CL-LNS** | **Gurobi** | **GNN\&GBDT** | **Light-MILPOpt** | **LLM-LNS (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dws` (min.) | 189028.6 | 186625.3 | -- | 146411.0 | * | 147417.7 | **143630.5** |
| `ivu` (min.) | 14261.3 | 9998.6 | -- | 27488.0 | * | -- | **3575.9** |